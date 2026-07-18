

#include "PrimaryGeneratorAction.hh"

#include <iomanip>

#include "DetectorConstruction.hh"
#include "Object.hh"
#include "Run.hh"
using namespace std;

#include "G4Event.hh"
#include "G4SystemOfUnits.hh"

#ifndef CRY_DATA
#define CRY_DATA "../data"
#endif

PrimaryGeneratorAction::PrimaryGeneratorAction(const char *inputfile)
{

  particleGun = new G4ParticleGun();

  std::ifstream inputFile;
  inputFile.open(inputfile, std::ios::in);
  char buffer[1000];

  if(inputFile.fail()) {
    if(*inputfile != 0) {
      G4cout << "PrimaryGeneratorAction: Failed to open CRY input file= " << inputfile << G4endl;
    }
    InputState = -1;
  } else {
    std::string setupString("");
    while(!inputFile.getline(buffer, 1000).eof()) {
      setupString.append(buffer);
      setupString.append(" ");
    }

    CRYSetup *setup = new CRYSetup(setupString, CRY_DATA);

    gen = new CRYGenerator(setup);

    RNGWrapper<CLHEP::HepRandomEngine>::set(CLHEP::HepRandom::getTheEngine(), &CLHEP::HepRandomEngine::flat);
    setup->setRandomFunction(RNGWrapper<CLHEP::HepRandomEngine>::rng);
    InputState = 0;
  }

  vect = new std::vector<CRYParticle *>;

  particleTable = G4ParticleTable::GetParticleTable();

  gunMessenger = new PrimaryGeneratorMessenger(this);

  fNPrimary = -1;
  fDetectorMaxZ = NAN;
  fDetectorHalfX = NAN;
  fDetectorHalfY = NAN;
}

PrimaryGeneratorAction::~PrimaryGeneratorAction() { }

void PrimaryGeneratorAction::Initialize(const DetectorConstruction *detectorConstruction)
{
  fDetectorMaxZ = detectorConstruction->GetMaxScoringZ();
  fDetectorHalfX = detectorConstruction->GetDetectorHalfX();
  fDetectorHalfY = detectorConstruction->GetDetectorHalfY();
}

void PrimaryGeneratorAction::InputCRY() { InputState = 1; }

void PrimaryGeneratorAction::UpdateCRY(std::string *MessInput)
{
  CRYSetup *setup = new CRYSetup(*MessInput, CRY_DATA);

  gen = new CRYGenerator(setup);

  RNGWrapper<CLHEP::HepRandomEngine>::set(CLHEP::HepRandom::getTheEngine(), &CLHEP::HepRandomEngine::flat);
  setup->setRandomFunction(RNGWrapper<CLHEP::HepRandomEngine>::rng);
  InputState = 0;
}

void PrimaryGeneratorAction::CRYFromFile(G4String newValue)
{

  std::ifstream inputFile;
  inputFile.open(newValue, std::ios::in);
  char buffer[1000];

  if(inputFile.fail()) {
    G4cout << "Failed to open input file " << newValue << G4endl;
    G4cout << "Make sure to define the cry library on the command line" << G4endl;
    InputState = -1;
  } else {
    std::string setupString("");
    while(!inputFile.getline(buffer, 1000).eof()) {
      setupString.append(buffer);
      setupString.append(" ");
    }

    CRYSetup *setup = new CRYSetup(setupString, CRY_DATA);

    gen = new CRYGenerator(setup);

    RNGWrapper<CLHEP::HepRandomEngine>::set(CLHEP::HepRandom::getTheEngine(), &CLHEP::HepRandomEngine::flat);
    setup->setRandomFunction(RNGWrapper<CLHEP::HepRandomEngine>::rng);
    InputState = 0;
  }
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event *anEvent)
{

  if(const char *bm = getenv("MUPOS_BEAM_MOMENTUM")) {
    G4double pbeam = std::atof(bm) * MeV;
    G4ParticleDefinition *mu = particleTable->FindParticle("mu-");
    G4double mass = mu->GetPDGMass();
    G4double ke = std::sqrt(pbeam * pbeam + mass * mass) - mass;
    Event *event = Run::GetInstance()->GetEvent();
    event->Reset();
    particleGun->SetParticleDefinition(mu);
    particleGun->SetParticleEnergy(ke);
    particleGun->SetParticlePosition({ fDetectorHalfX * (2 * G4UniformRand() - 1),
                                       fDetectorHalfY * (2 * G4UniformRand() - 1),
                                       fDetectorMaxZ });
    particleGun->SetParticleMomentumDirection({ 0.0, 0.0, -1.0 });
    particleGun->SetParticleTime(0);
    particleGun->GeneratePrimaryVertex(anEvent);
    fNPrimary = 1;
    event->Pid = mu->GetPDGEncoding();
    event->Px = 0.0; event->Py = 0.0; event->Pz = -pbeam; event->E = ke + mass;
    G4ThreeVector vp = particleGun->GetParticlePosition();
    event->X = vp.x(); event->Y = vp.y(); event->Z = vp.z(); event->T = 0.0;
    return;
  }

  if(InputState != 0) {
    G4String *str = new G4String("CRY library was not successfully initialized");

    G4Exception("PrimaryGeneratorAction", "1", RunMustBeAborted, *str);
  }
  vect->clear();
  gen->genEvent(vect);

  Event *event = Run::GetInstance()->GetEvent();
  event->Reset();
  if(__builtin_expect(vect->empty(), false)) return;
  fNPrimary = 0;
  for(unsigned j = 0, j0 = G4UniformRand() * vect->size(); j < vect->size(); j++) {

    if(j == j0) {
      particleGun->SetParticleDefinition(particleTable->FindParticle((*vect)[j]->PDGid()));
      particleGun->SetParticleEnergy((*vect)[j]->ke() * MeV);

      particleGun->SetParticlePosition({
          fDetectorHalfX * (2 * G4UniformRand() - 1),
          fDetectorHalfY * (2 * G4UniformRand() - 1),
          fDetectorMaxZ,
      });
      particleGun->SetParticleMomentumDirection(G4ThreeVector((*vect)[j]->u(), (*vect)[j]->v(), (*vect)[j]->w()));

      particleGun->SetParticleTime(0);
      particleGun->GeneratePrimaryVertex(anEvent);
      ++fNPrimary;
      G4double mass = particleGun->GetParticleDefinition()->GetPDGMass(), e = particleGun->GetParticleEnergy() + mass;
      event->Pid = particleGun->GetParticleDefinition()->GetPDGEncoding();
      G4ThreeVector v = sqrt(e*e - mass*mass) * particleGun->GetParticleMomentumDirection();
      event->Px = v.x();
      event->Py = v.y();
      event->Pz = v.z();
      event->E = e;
      v = particleGun->GetParticlePosition();
      event->X = v.x();
      event->Y = v.y();
      event->Z = v.z();
      event->T = particleGun->GetParticleTime();
    }
    delete(*vect)[j];
  }
}
