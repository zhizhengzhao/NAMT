

#include "Run.hh"

#include <TClonesArray.h>
#include <TFile.h>
#include <TTree.h>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <syscall.h>
#include <unistd.h>

#include <filesystem>

#include "DetectorConstruction.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VProcess.hh"
#include "G4ios.hh"
#include "Object.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunMessenger.hh"

namespace {

constexpr double GRID_VOX = 20.0;
constexpr double GRID_XMIN = -200.0, GRID_YMIN = -200.0, GRID_ZMIN = -200.0;
constexpr int GRID_NX = 20, GRID_NY = 20, GRID_NZ = 20;
inline int gridIndex(double x, double y, double z)
{
  int ix = (int)((x - GRID_XMIN) / GRID_VOX);
  int iy = (int)((y - GRID_YMIN) / GRID_VOX);
  int iz = (int)((z - GRID_ZMIN) / GRID_VOX);
  if(ix < 0 || ix >= GRID_NX || iy < 0 || iy >= GRID_NY || iz < 0 || iz >= GRID_NZ) return -1;
  return (ix * GRID_NY + iy) * GRID_NZ + iz;
}
}

Run::Run()
{
  fRunMessenger = new RunMessenger(this);
  fPrimaryGeneratorAction = (PrimaryGeneratorAction *)G4RunManager::GetRunManager()->GetUserPrimaryGeneratorAction();
  fDetectorConstruction = (DetectorConstruction *)G4RunManager::GetRunManager()->GetUserDetectorConstruction();
  fRootName = "CryMu.root";
  fTree = NULL;
  fFile = NULL;
  fIEvent = 0;
}

Run::~Run()
{
  SaveTree();
  delete fRunMessenger;
}

Run *Run::GetInstance()
{
  static Run run;
  return &run;
}

void Run::InitGeom()
{
  G4double scoringHalfZ = fDetectorConstruction->GetScoringHalfZ();
  const std::vector<G4double> &scoringZs = fDetectorConstruction->GetScoringZs();

  fScoringHalfX = fDetectorConstruction->GetScoringHalfX();
  fScoringHalfY = fDetectorConstruction->GetScoringHalfY();
  fScoringZ = scoringHalfZ * 2;
  fScoringMaxZs = scoringZs;
  for(G4double &z : fScoringMaxZs) z += scoringHalfZ;
  fStatus.resize(fScoringMaxZs.size());
}

void Run::InitTree()
{
  using namespace std::filesystem;
  auto dirpath = path(fRootName.c_str()).parent_path();
  if(!dirpath.empty()) { create_directories(dirpath); }

  fFile = TFile::Open(fRootName, "RECREATE");

  fTree = new TTree("tree", "tree");
  fTree->Branch("Tracks", new TClonesArray("Track"));
  fTree->Branch("Edeps", new TClonesArray("Edep"));
  fTree->Branch("EdepGrid", new TClonesArray("Edep"));
  fTree->Branch("Event", new TClonesArray("Event"));
  (*(TClonesArray **)fTree->GetBranch("Event")->GetAddress())->ConstructedAt(0);

  fParams = new TTree("params", "params");
  fParams->Branch("Params", new TClonesArray("Params"));
  (*(TClonesArray **)fParams->GetBranch("Params")->GetAddress())->ConstructedAt(0);
  fParams->Branch("Processes", new TClonesArray("Process"));

  BuildProcessMap();
}

void Run::SaveTree()
{
  if(!fFile) { return; }
  fFile->cd();

  fTree->Write(NULL, TObject::kOverwrite);
  delete *(TClonesArray **)fTree->GetBranch("Tracks")->GetAddress();
  delete *(TClonesArray **)fTree->GetBranch("Edeps")->GetAddress();
  delete *(TClonesArray **)fTree->GetBranch("EdepGrid")->GetAddress();
  delete *(TClonesArray **)fTree->GetBranch("Event")->GetAddress();
  fTree = NULL;

  Params *params = (Params *)(*(TClonesArray **)fParams->GetBranch("Params")->GetAddress())->At(0);
  params->NEvent = fIEvent;
  *params = *fDetectorConstruction;
  TClonesArray *Processes = *(TClonesArray **)fParams->GetBranch("Processes")->GetAddress();
  for(auto &[name, id] : fProcessMap) *(::Process *)Processes->ConstructedAt(Processes->GetEntries()) = { id, name };
  fParams->Fill();
  fParams->Write(NULL, TObject::kOverwrite);
  delete *(TClonesArray **)fParams->GetBranch("Params")->GetAddress();
  delete *(TClonesArray **)fParams->GetBranch("Processes")->GetAddress();
  fParams = NULL;

  fFile->Close();
  fFile = NULL;
}

void Run::FillAndReset()
{
  auto Tracks = *(TClonesArray **)fTree->GetBranch("Tracks")->GetAddress();
  auto Edeps = *(TClonesArray **)fTree->GetBranch("Edeps")->GetAddress();

  for(Int_t i = Tracks->GetEntries() - 1; i >= 0; --i) {
    Track *track = (Track *)(*Tracks)[i];
    G4double x = track->X, y = track->Y;
    if(fabs(x) > fScoringHalfX || fabs(y) > fScoringHalfY) Tracks->RemoveAt(i);
  }
  Tracks->Compress();

  std::vector<Track *> tracks;
  tracks.resize(Tracks->GetEntries());
  for(size_t i = 0; i < tracks.size(); ++i) tracks[i] = (Track *)(*Tracks)[i];
  sort(tracks.begin(), tracks.end(), [](Track *a, Track *b) { return a->Id < b->Id; });
  for(size_t i = 0; i < tracks.size(); ++i) (*Tracks)[i] = tracks[i];

  if(all_of(fStatus.begin(), fStatus.end(), [](bool b) { return b; })) {
    for(auto &edep : fEdep) { *(::Edep *)Edeps->ConstructedAt(Edeps->GetEntries()) = edep; }
    auto EdepGrid = *(TClonesArray **)fTree->GetBranch("EdepGrid")->GetAddress();
    for(auto &[gidx, energy] : fGrid) {
      if(energy <= 1e-4) continue;
      int iz = gidx % GRID_NZ;
      int iy = (gidx / GRID_NZ) % GRID_NY;
      int ix = gidx / (GRID_NZ * GRID_NY);
      ::Edep *e = (::Edep *)EdepGrid->ConstructedAt(EdepGrid->GetEntries());
      e->Id = iz;
      e->Pid = ix;
      e->Process = iy;
      e->trackID = 0;
      e->Value = energy;
      e->X = GRID_XMIN + (ix + 0.5) * GRID_VOX;
      e->Y = GRID_YMIN + (iy + 0.5) * GRID_VOX;
      e->T = 0.0;
    }
    fTree->Fill();
    Edeps->Clear();
    EdepGrid->Clear();
  }
  fStatus.assign(fStatus.size(), false);

  Tracks->Clear();
  fEdep.clear();
  fGrid.clear();
  ++fIEvent;
}

void Run::AutoSave() { fTree->AutoSave("SaveSelf Overwrite"); }

void Run::AddStep(const G4Step *step)
{

  if(step->GetTrack()->GetTrackID() == 1) {
    if(const G4VPhysicalVolume *pv = step->GetPreStepPoint()->GetPhysicalVolume()) {
      if(pv->GetName().find("water_sphere") != std::string::npos) {
        Event *ev = GetEvent();
        ev->ThroughWater = 1;
        ev->WaterPath += step->GetStepLength();
      }
    }
  }

  const G4ThreeVector &r = step->GetTrack()->GetPosition();
  G4double x = r.x(), y = r.y(), z = r.z();
  G4double edep = step->GetTotalEnergyDeposit();
  G4double time = step->GetTrack()->GetGlobalTime();

  if(edep > 1e-5) {
    int gidx = gridIndex(x, y, z);
    if(gidx >= 0) fGrid[gidx] += edep;
  }

  if(fabs(x) >= fScoringHalfX || fabs(y) >= fScoringHalfY) return;
  auto ub = std::upper_bound(fScoringMaxZs.begin(), fScoringMaxZs.end(), z);
  if(ub == fScoringMaxZs.end()) return;
  if(z < *ub - fScoringZ) return;
  if(edep <= 1e-6) return;

  Int_t zid = fScoringMaxZs.size() - 1 - (ub - fScoringMaxZs.begin());
  Int_t pid = (uint32_t)step->GetTrack()->GetParticleDefinition()->GetPDGEncoding();
  Int_t process = -1;
  Int_t trackid = step->GetTrack()->GetTrackID();
  if(const G4VProcess *p = step->GetTrack()->GetCreatorProcess()) {
    auto it = fProcessMap.find(p->GetProcessName());
    if(it != fProcessMap.end()) process = it->second;
  }
  fStatus[zid] = true;
  fEdep[{ zid, pid, process, trackid }].Add(edep, x, y, time, trackid);
}

void Run::AddTrack([[maybe_unused]] const G4Track *track)
{
  auto Tracks = *(TClonesArray **)fTree->GetBranch("Tracks")->GetAddress();
  *(Track *)Tracks->ConstructedAt(Tracks->GetEntries()) = *track;
}

void Run::BuildProcessMap()
{
  auto &iter = *G4ParticleTable::GetParticleTable()->GetIterator();
  iter.reset();
  while(iter()) {
    G4ParticleDefinition *particle = iter.value();
    G4ProcessManager *processManager = particle->GetProcessManager();
    if(!processManager) continue;
    G4ProcessVector *processList = processManager->GetProcessList();
    if(!processList) continue;
    for(size_t i = 0; i < processList->size(); ++i) {
      G4VProcess *process = (*processList)[i];
      fProcessMap[process->GetProcessName()] = 0;
    }
  }
  size_t i = 0;
  for(auto &[name, id] : fProcessMap) {
    id = i++;
    G4cout << __PRETTY_FUNCTION__ << ": " << std::setw(3) << id << " " << name << G4endl;
  }
}

Event *Run::GetEvent() { return (Event *)(*(TClonesArray **)fTree->GetBranch("Event")->GetAddress())->At(0); }

uint64_t Run::GetThreadId()
{
#ifdef __APPLE__
  uint64_t tid;
  pthread_threadid_np(NULL, &tid);
  return tid;
#else
  int64_t tid = syscall(SYS_gettid);
  if(tid < 0) {
    perror("gettid");
    exit(EXIT_FAILURE);
  }
  return tid;
#endif
}

uint64_t Run::GetSeed()
{

  if(const char *configured = std::getenv("MUPOS_SEED")) {
    char *end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(configured, &end, 10);
    if(errno == 0 && end != configured && *end == '\0') {
      return static_cast<uint64_t>(parsed);
    }
    G4cerr << "WARNING: invalid MUPOS_SEED='" << configured
           << "'; falling back to a time-based seed." << G4endl;
  }
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count()
      + GetThreadId();
}
