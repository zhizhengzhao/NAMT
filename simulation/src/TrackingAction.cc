

#include "TrackingAction.hh"

#include "Run.hh"

TrackingAction::TrackingAction() { }

TrackingAction::~TrackingAction() { }

void TrackingAction::PreUserTrackingAction([[maybe_unused]] const G4Track *track)
{
  Run::GetInstance()->AddTrack(track);

}

void TrackingAction::PostUserTrackingAction([[maybe_unused]] const G4Track *track) { }
