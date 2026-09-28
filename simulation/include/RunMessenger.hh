

#ifndef RunMessenger_h
#define RunMessenger_h 1

#include "G4UImessenger.hh"

class Run;
class G4UIdirectory;
class G4UIcmdWithAString;

class RunMessenger : public G4UImessenger {
public:
  RunMessenger(Run *run);
  ~RunMessenger() override;
  void SetNewValue(G4UIcommand *, G4String) override;

private:
  Run *fRun;
  G4UIdirectory *fFileNameDir;
  G4UIcmdWithAString *fSetFileNameCmd;

  class Driver;
  class Driver *fDriver;
};

#endif
