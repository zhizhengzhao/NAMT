

#include "RunAction.hh"

#include "Run.hh"

RunAction::RunAction() { Run::GetInstance(); }

RunAction::~RunAction() { }

void RunAction::BeginOfRunAction(const G4Run *)
{
  Run::GetInstance()->InitGeom();
  Run::GetInstance()->InitTree();
}

void RunAction::EndOfRunAction(const G4Run *) { Run::GetInstance()->SaveTree(); }

Run *RunAction::GetRun() const { return Run::GetInstance(); }
