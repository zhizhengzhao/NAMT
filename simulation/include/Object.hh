

#ifndef Object_h
#define Object_h 1

#include <TObject.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "EdepData.hh"

class DetectorConstruction;
class G4Track;
class G4DynamicParticle;

class Track : public TObject {
public:
  Track &operator=(const G4Track &);

  Int_t Id;
  Int_t Mother;
  Int_t Pid;
  Double_t Px;
  Double_t Py;
  Double_t Pz;
  Double_t E;
  Double_t X;
  Double_t Y;
  Double_t Z;
  Double_t T;

  ClassDef(Track, 1);
};

class Params : public TObject {
public:
  Params &operator=(const DetectorConstruction &);

  Long64_t NEvent;
  Double_t GammaCut;
  Double_t GammaThreshold;
  Double_t ElectronCut;
  Double_t ElectronThreshold;
  Double_t PositronCut;
  Double_t PositronThreshold;
  Double_t ProtonCut;
  Double_t ProtonThreshold;

  std::vector<double> LayerZ;

  ClassDef(Params, 1);
};

class Edep : public TObject {
public:
  Edep &operator=(std::pair<const EdepKey, EdepValue> &p)
  {
    auto &[key, value] = p;
    std::tie(Id, Pid, Process, trackID) = key.Tuple();
    std::tie(Value, X, Y, T, trackID) = value.Finish().Tuple();
    return *this;
  }

  Int_t Id;
  Int_t Pid;
  Int_t Process;
  Int_t trackID;
  Double_t Value;
  Double_t X;
  Double_t Y;
  Double_t T;

  ClassDef(Edep, 1);
};

class Process : public TObject {
public:
  Process &operator=(const std::pair<Int_t, const std::string &> &t)
  {
    std::tie(Id, Name) = t;
    return *this;
  }

  Int_t Id;
  std::string Name;

  ClassDef(Process, 1);
};
class Event : public TObject {
  public:
    Int_t Pid;
    Double_t Px;
    Double_t Py;
    Double_t Pz;
    Double_t E;
    Double_t X;
    Double_t Y;
    Double_t Z;
    Double_t T;
    Int_t ThroughWater;
    Double_t WaterPath;

    void Reset() { memset(&Pid, 0, (char *)&WaterPath - (char *)&Pid + sizeof WaterPath); }

    ClassDef(Event, 3);
};

#endif
