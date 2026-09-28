

#include "Object.hh"

#include <math.h>

#include "DetectorConstruction.hh"
#include "G4LogicalVolume.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ProductionCuts.hh"
#include "G4RToEConvForElectron.hh"
#include "G4RToEConvForGamma.hh"
#include "G4RToEConvForPositron.hh"
#include "G4RToEConvForProton.hh"
#include "G4Track.hh"

Track &Track::operator=(const G4Track &track)
{
  auto position = track.GetPosition();
  auto momentum = track.GetMomentum();

  Id = track.GetTrackID();
  Mother = track.GetParentID();
  Pid = track.GetParticleDefinition()->GetPDGEncoding();
  Px = momentum.getX();
  Py = momentum.getY();
  Pz = momentum.getZ();
  E = track.GetTotalEnergy();
  X = position.getX();
  Y = position.getY();
  Z = position.getZ();
  T = track.GetGlobalTime();

  return *this;
}

Params &Params::operator=(const DetectorConstruction &detectorConstruction)
{
  const G4MaterialCutsCouple *couple = detectorConstruction.GetScoringGasVolume()->GetMaterialCutsCouple();
  const G4Material *material = couple->GetMaterial();
  G4ProductionCuts *cuts = couple->GetProductionCuts();

  GammaCut = cuts->GetProductionCut("gamma");
  ElectronCut = cuts->GetProductionCut("e-");
  PositronCut = cuts->GetProductionCut("e+");
  ProtonCut = cuts->GetProductionCut("proton");

  GammaThreshold = G4RToEConvForGamma().Convert(GammaCut, material);
  ElectronThreshold = G4RToEConvForElectron().Convert(ElectronCut, material);
  PositronThreshold = G4RToEConvForPositron().Convert(PositronCut, material);
  ProtonThreshold = G4RToEConvForProton().Convert(ProtonCut, material);

  LayerZ = detectorConstruction.GetScoringZs();
  std::sort(LayerZ.begin(), LayerZ.end(), std::greater<G4double>());

  return *this;
}
