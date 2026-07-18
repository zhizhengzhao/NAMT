

#include "SteppingAction.hh"

#include "Run.hh"

SteppingAction::SteppingAction() { }

SteppingAction::~SteppingAction() { }

void SteppingAction::UserSteppingAction(const G4Step *step) { Run::GetInstance()->AddStep(step); }
