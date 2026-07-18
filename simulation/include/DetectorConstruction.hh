

#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include <functional>
#include <vector>

#include "G4RotationMatrix.hh"
#include "G4ThreeVector.hh"
#include "G4VUserDetectorConstruction.hh"

class G4VSolid;
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4LogicalVolumeStore;
class G4PhysicalVolumeStore;

#define DETECTOR_OPTION_SCORING_ONLY 0b00000001
#define DETECTOR_OPTION_VACUUM_ENV   0b00000010

class DetectorConstruction : public G4VUserDetectorConstruction {
public:
  DetectorConstruction(int options = 0);
  ~DetectorConstruction() override;

  G4VPhysicalVolume *Construct() override;

  G4double GetScoringHalfX() const { return fElectrodeHalfX; }
  G4double GetScoringHalfY() const { return fElectrodeHalfY; }
  G4double GetScoringHalfZ() const { return fScoringHalfZ; }
  const std::vector<G4double> &GetScoringZs() const { return fScoringZs; }
  G4double GetMaxScoringZ() const;
  G4double GetDetectorHalfX() const;
  G4double GetDetectorHalfY() const;
  G4LogicalVolume *GetScoringGasVolume() const { return fScoringGasVolume; }
  const std::vector<G4LogicalVolume *> &GetDetPanels() const { return fDetPanels; }

  void PrintVolumes(G4VPhysicalVolume *) const;
  void WalkVolume(G4LogicalVolume *volume, const std::function<void(G4LogicalVolume *)> &enter,
      const std::function<void(G4LogicalVolume *)> &leave = nullptr) const;
  void WalkVolume(G4VPhysicalVolume *volume, const std::function<void(G4VPhysicalVolume *)> &enter,
      const std::function<void(G4VPhysicalVolume *)> &leave = nullptr) const;
  void WalkVolume(G4VPhysicalVolume *volume,
      const std::function<void(G4VPhysicalVolume *, const G4ThreeVector &, const G4RotationMatrix &)> &enter,
      const std::function<void(G4VPhysicalVolume *, const G4ThreeVector &, const G4RotationMatrix &)> &leave =
          nullptr) const;
  G4VPhysicalVolume *PartitionVolume(G4VPhysicalVolume *volume,
      const std::function<std::vector<G4VSolid *>(G4VSolid *, const G4ThreeVector &, const G4RotationMatrix &)>
          &partition) const;

private:
  void DefineMaterials();
  void DefineVolumes();
  void DefineFields();

  const int fOptions;
  G4LogicalVolumeStore *fLogicalVolumeStore;
  G4PhysicalVolumeStore *fPhysicalVolumeStore;
  G4VPhysicalVolume *fWorld;
  G4double fElectrodeHalfX, fElectrodeHalfY, fElectrodeHalfZ, fScoringHalfZ;
  std::vector<G4double> fElectrodeZs;
  std::vector<G4double> fScoringZs;
  G4double fMaxScoringZ;
  G4LogicalVolume *fScoringGasVolume;
  std::vector<G4LogicalVolume *> fDetPanels;
};

#endif
