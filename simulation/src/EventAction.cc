

#include "EventAction.hh"

#include "G4Event.hh"
#include "Run.hh"

EventAction::EventAction() { }

EventAction::~EventAction() { }

void EventAction::BeginOfEventAction(const G4Event *) { }

void EventAction::EndOfEventAction(const G4Event *evt)
{

  Run::GetInstance()->FillAndReset();

  G4int event_id = evt->GetEventID();
  if((event_id + 1) % 10000 == 0) { Run::GetInstance()->AutoSave(); }
}
