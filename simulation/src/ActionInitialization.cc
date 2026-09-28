

#include "ActionInitialization.hh"

#include "EventAction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "SteppingAction.hh"
#include "TrackingAction.hh"

void ActionInitialization::Build() const
{
  PrimaryGeneratorAction *primary = new PrimaryGeneratorAction("");
  SetUserAction(primary);

  RunAction *runAction = new RunAction;
  SetUserAction(runAction);

  EventAction *eventAction = new EventAction;
  SetUserAction(eventAction);

  SteppingAction *stepAction = new SteppingAction;
  SetUserAction(stepAction);

  TrackingAction *trackAction = new TrackingAction;
  SetUserAction(trackAction);
}
