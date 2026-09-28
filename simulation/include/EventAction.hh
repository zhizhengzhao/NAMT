

#ifndef EventAction_h
#define EventAction_h 1

#include "G4UserEventAction.hh"

class EventAction : public G4UserEventAction {
public:
  EventAction();
  ~EventAction() override;

  void BeginOfEventAction(const G4Event *) override;
  void EndOfEventAction(const G4Event *) override;
};

#endif
