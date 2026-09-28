

#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "FTFP_BERT.hh"
#include "G4AutoDelete.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4RunManager.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4UIExecutive.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#include "Run.hh"

int main(int argc, char **argv)
{

  G4UIExecutive *ui = NULL;
  if(argc == 1) ui = new G4UIExecutive(argc, argv);

  auto engine = new CLHEP::RanecuEngine;
  G4Random::setTheEngine(engine);
  G4AutoDelete::Register(engine);
  uint64_t seed = Run::GetSeed();
  G4cout << "seed=" << seed << G4endl;
  G4Random::setTheSeed(seed);

  G4RunManager *runManager = new G4RunManager;

  runManager->SetUserInitialization(new DetectorConstruction);
  G4VModularPhysicsList *physicsList = new FTFP_BERT;
  physicsList->ReplacePhysics(new G4EmStandardPhysics_option4());
  physicsList->RegisterPhysics(new G4StepLimiterPhysics());
  runManager->SetUserInitialization(physicsList);
  runManager->SetUserInitialization(new ActionInitialization);

  runManager->Initialize();

  G4VisManager *visManager = nullptr;

  G4UImanager *UImanager = G4UImanager::GetUIpointer();

  if(ui) {
    visManager = new G4VisExecutive;
    visManager->Initialize();
    UImanager->ApplyCommand("/control/execute vis.mac");
    ui->SessionStart();
    delete ui;
  } else {
    G4String command = "/control/execute ";
    G4String fileName = argv[1];
    UImanager->ApplyCommand(command + fileName);
  }

  delete visManager;
  delete runManager;
}
