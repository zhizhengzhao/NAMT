

#include "DetectorConstruction.hh"

#include "G4AutoDelete.hh"
#include "G4RunManager.hh"
#include "GeometryConfig.hh"
#include "PrimaryGeneratorAction.hh"

#include "G4Box.hh"
#include "G4GeometryManager.hh"
#include "G4IntersectionSolid.hh"
#include "G4Orb.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4PVPlacement.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4SolidStore.hh"
#include "G4SubtractionSolid.hh"
#include "G4Tubs.hh"
#include "G4ExtrudedSolid.hh"
#include "G4UnionSolid.hh"
#include "G4EllipticalTube.hh"
#include "G4TwoVector.hh"
#include "G4NistManager.hh"
#include <cstdlib>
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include <cmath>
#include <string>
#include <fstream>
#include <algorithm>

#include "G4Color.hh"
#include "G4VisAttributes.hh"

#include "G4ChordFinder.hh"
#include "G4ClassicalRK4.hh"
#include "G4EqMagElectricField.hh"
#include "G4FieldManager.hh"
#include "G4MagIntegratorDriver.hh"
#include "G4UniformElectricField.hh"
#include "G4UniformMagField.hh"
#include "G4UserLimits.hh"

DetectorConstruction::DetectorConstruction(int o)
    : fOptions(o),
      fWorld(NULL),
      fElectrodeHalfX(0.0),
      fElectrodeHalfY(0.0),
      fElectrodeHalfZ(0.0),
      fScoringHalfZ(0.0),
      fMaxScoringZ(0.0),
      fScoringGasVolume(NULL)
{
  if(fOptions) { throw std::invalid_argument("options unimplemented"); }
  fLogicalVolumeStore = G4LogicalVolumeStore::GetInstance();
  fPhysicalVolumeStore = G4PhysicalVolumeStore::GetInstance();
}

DetectorConstruction::~DetectorConstruction()
{

}

static std::vector<std::string> split(const std::string &str, char c)
{
  std::vector<std::string> v;
  size_t p = 0, q = 0;
  while((q = str.find(c, p)) != str.npos) {
    v.push_back(str.substr(p, q - p));
    p = q + 1;
  }
  v.push_back(str.substr(p));
  return v;
}

void DetectorConstruction::DefineMaterials()
{
  std::vector<std::string> paths = {
    "../config/newrpc_material.yaml",
  };
  const char *matExtra = getenv("MUPOS_EXTRA_MATERIAL");
  if(matExtra) {
    for(const auto &ep : split(matExtra, ':')) {
      GeometryConfig::LoadMaterials(ep.c_str());
    }
  }
  char *p = getenv("MUPOS_MATERIAL_CONFIG");
  if(p) { paths = split(p, ':'); }
  for(const std::string &path : paths) { GeometryConfig::LoadMaterials(path.c_str()); }
}

void DetectorConstruction::DefineVolumes()
{
  std::vector<std::string> paths = {
    "../config/newrpc_readout.yaml",
    "../config/newrpc.yaml",
    "../config/newlayout.yaml",
  };
  char *p = getenv("MUPOS_VOLUME_CONFIG");
  if(p) { paths = split(p, ':'); }
  for(const std::string &path : paths) { GeometryConfig::LoadVolumes(path.c_str()); }

  fWorld = new G4PVPlacement(0, { 0, 0, 0 }, fLogicalVolumeStore->GetVolume("world"), "world", 0, false, 0, true);
  G4LogicalVolume *newrpc_electrode = fLogicalVolumeStore->GetVolume("newrpc_electrode");
  fElectrodeHalfX = dynamic_cast<G4Box *>(newrpc_electrode->GetSolid())->GetXHalfLength();
  fElectrodeHalfY = dynamic_cast<G4Box *>(newrpc_electrode->GetSolid())->GetYHalfLength();
  fElectrodeHalfZ = dynamic_cast<G4Box *>(newrpc_electrode->GetSolid())->GetZHalfLength();
  fElectrodeZs.assign(0, 0.0);
  WalkVolume(
      fWorld, [newrpc_electrode, this](G4VPhysicalVolume *volume, const G4ThreeVector &r, const G4RotationMatrix &) {
        if(volume->GetLogicalVolume() != newrpc_electrode) { return; }
        fElectrodeZs.push_back(r.z());
      });
  sort(fElectrodeZs.begin(), fElectrodeZs.end());
  fScoringHalfZ = (fElectrodeZs.at(1) -fElectrodeZs.at(0)) * 0.5 - fElectrodeHalfZ;

  const std::string scoringReference = getenv("MUPOS_SCORING_REFERENCE")
      ? getenv("MUPOS_SCORING_REFERENCE") : "module_center";
  if(scoringReference == "module_center") {
    G4LogicalVolume *newrpc = fLogicalVolumeStore->GetVolume("newrpc");
    WalkVolume(
        fWorld, [newrpc, this](G4VPhysicalVolume *volume, const G4ThreeVector &r, const G4RotationMatrix &) {
          if(volume->GetLogicalVolume() != newrpc) { return; }
          fScoringZs.push_back(r.z());
        });
  } else if(scoringReference == "readout_board") {
    G4LogicalVolume *board = fLogicalVolumeStore->GetVolume("newrpc_xy_readout_board");
    WalkVolume(
        fWorld, [board, this](G4VPhysicalVolume *volume, const G4ThreeVector &r, const G4RotationMatrix &) {
          if(volume->GetLogicalVolume() != board) { return; }
          fScoringZs.push_back(r.z());
        });
  } else {
    G4cerr << "ERROR: MUPOS_SCORING_REFERENCE must be 'module_center' or 'readout_board', got '"
           << scoringReference << "'." << G4endl;
    exit(EXIT_FAILURE);
  }
  std::sort(fScoringZs.begin(), fScoringZs.end());
  G4cout << "[geometry] scoring reference: " << scoringReference << G4endl;
  fMaxScoringZ = *std::max_element(fScoringZs.begin(), fScoringZs.end()) + fScoringHalfZ;
  fScoringGasVolume = fLogicalVolumeStore->GetVolume("newrpc_gas");

  const char *prefixes[4] = {"POTATO", "POTATO2", "POTATO3", "POTATO4"};
  for(int iObj = 0; iObj < 4; ++iObj) {
  const std::string pfx = prefixes[iObj];
  auto envs = [&pfx](const char *key) -> const char * {
    std::string k = pfx + "_" + key;
    return getenv(k.c_str());
  };
  const char *px = envs("X");
  const char *py = envs("Y");
  const char *pz = envs("Z");
  const char *pr = envs("R");
  if(px && py && pz) {
    const char *pmat = envs("MAT");
    G4Material *objMat = pmat
        ? G4NistManager::Instance()->FindOrBuildMaterial(pmat)
        : G4Material::GetMaterial("potato_water", false);
    if(!objMat && !pmat) objMat = G4Material::GetMaterial("G4_WATER", false);
    if(!objMat) {
      G4cerr << "ERROR: object material (" << (pmat ? pmat : "potato_water")
             << ") not found." << G4endl;
      exit(EXIT_FAILURE);
    }
    G4LogicalVolume *gapLogical = fLogicalVolumeStore->GetVolume("world_major_gap");
    if(!gapLogical) {
      G4cerr << "ERROR: world_major_gap volume not found." << G4endl;
      exit(EXIT_FAILURE);
    }
    auto gapBox = dynamic_cast<G4Box *>(gapLogical->GetSolid());

    G4double cx = std::atof(px) * mm;
    G4double cy = std::atof(py) * mm;
    G4double cz = std::atof(pz) * mm;
    G4double R = (pr ? std::atof(pr) : 60.0) * mm;
    std::string shape = envs("SHAPE") ? envs("SHAPE") : "orb";
    G4double ang = envs("ANG") ? std::atof(envs("ANG")) * deg : 0.0;
    const std::string solidName = "water_sphere_" + std::to_string(iObj);

    const G4double hzFull = (gapBox ? gapBox->GetZHalfLength() - 0.5 * mm : 234.5 * mm);
    const G4double hz = envs("HZ") ? std::atof(envs("HZ")) * mm : hzFull;

    G4VSolid *shapeSolid = nullptr;
    if(shape == "orb") {
      shapeSolid = new G4Orb(solidName + "_orb", R);
    } else if(shape == "circle") {
      shapeSolid = new G4Tubs(solidName + "_cyl", 0.0, R, hz, 0.0, twopi);
    } else if(shape == "square") {

      std::vector<G4TwoVector> poly;
      for(int k = 0; k < 4; ++k) {
        G4double a = pi / 4.0 + ang - k * halfpi;
        poly.emplace_back(R * std::sqrt(2.0) * std::cos(a), R * std::sqrt(2.0) * std::sin(a));
      }
      shapeSolid = new G4ExtrudedSolid(solidName + "_sq", poly, hz,
                                       G4TwoVector(0, 0), 1.0, G4TwoVector(0, 0), 1.0);
    } else if(shape == "triangle") {

      std::vector<G4TwoVector> poly;
      for(int k = 0; k < 3; ++k) {
        G4double a = halfpi + ang - k * twopi / 3.0;
        poly.emplace_back(R * std::cos(a), R * std::sin(a));
      }
      shapeSolid = new G4ExtrudedSolid(solidName + "_tri", poly, hz,
                                       G4TwoVector(0, 0), 1.0, G4TwoVector(0, 0), 1.0);
    } else if(shape == "rect") {

      G4double rx = envs("RX") ? std::atof(envs("RX")) * mm : R;
      G4double ry = envs("RY") ? std::atof(envs("RY")) * mm : 0.6 * R;
      G4double cs = std::cos(ang), sn = std::sin(ang);
      G4double xs[4] = { rx, rx, -rx, -rx }, ys[4] = { ry, -ry, -ry, ry };
      std::vector<G4TwoVector> poly;
      for(int k = 0; k < 4; ++k) poly.emplace_back(xs[k] * cs - ys[k] * sn, xs[k] * sn + ys[k] * cs);
      shapeSolid = new G4ExtrudedSolid(solidName + "_rect", poly, hz,
                                       G4TwoVector(0, 0), 1.0, G4TwoVector(0, 0), 1.0);
    } else if(shape == "ellipse") {

      G4double rx = envs("RX") ? std::atof(envs("RX")) * mm : R;
      G4double ry = envs("RY") ? std::atof(envs("RY")) * mm : 0.6 * R;
      shapeSolid = new G4EllipticalTube(solidName + "_ell", rx, ry, hz);
    } else if(shape == "ngon") {

      int n = envs("NSIDES") ? std::atoi(envs("NSIDES")) : 5;
      if(n < 3) n = 3;
      std::vector<G4TwoVector> poly;
      for(int k = 0; k < n; ++k) {
        G4double a = halfpi + ang - k * twopi / n;
        poly.emplace_back(R * std::cos(a), R * std::sin(a));
      }
      shapeSolid = new G4ExtrudedSolid(solidName + "_ngon", poly, hz,
                                       G4TwoVector(0, 0), 1.0, G4TwoVector(0, 0), 1.0);
    } else if(shape == "pshape") {

      auto sc = [&](double X, double Y) { return G4TwoVector(X * R, Y * R); };
      std::vector<G4TwoVector> outer = {
        sc(-0.50, -1.00), sc(-0.50, 1.00), sc(0.18, 1.00), sc(0.42, 0.90),
        sc(0.56, 0.62), sc(0.56, 0.30), sc(0.42, 0.06), sc(0.16, -0.05),
        sc(-0.16, -0.05), sc(-0.16, -1.00) };
      std::vector<G4TwoVector> hole = {
        sc(-0.06, 0.28), sc(-0.06, 0.78), sc(0.16, 0.78), sc(0.34, 0.64),
        sc(0.40, 0.46), sc(0.34, 0.28), sc(0.16, 0.18), sc(-0.06, 0.18) };
      auto pout = new G4ExtrudedSolid(solidName + "_Pout", outer, hz,
                                      G4TwoVector(0, 0), 1.0, G4TwoVector(0, 0), 1.0);
      auto phole = new G4ExtrudedSolid(solidName + "_Phole", hole, hz * 1.02,
                                       G4TwoVector(0, 0), 1.0, G4TwoVector(0, 0), 1.0);
      shapeSolid = new G4SubtractionSolid(solidName + "_P", pout, phole);
    } else if(shape == "polyfile") {

      const char *pf = envs("POLYFILE");
      if(!pf) { G4cerr << "ERROR: POTATO_POLYFILE not set." << G4endl; exit(EXIT_FAILURE); }
      std::ifstream in(pf);
      if(!in) { G4cerr << "ERROR: cannot open polyfile " << pf << G4endl; exit(EXIT_FAILURE); }
      std::string kind; int n; int idx = 0;
      G4VSolid *acc = nullptr; std::vector<G4VSolid *> holes;
      while(in >> kind >> n) {
        std::vector<G4TwoVector> poly; poly.reserve(n);
        for(int k = 0; k < n; ++k) { double xx, yy; in >> xx >> yy; poly.emplace_back(xx * mm, yy * mm); }

        double area = 0; for(int k = 0; k + 1 < (int)poly.size(); ++k) area += poly[k].x() * poly[k+1].y() - poly[k+1].x() * poly[k].y();
        if(area > 0) std::reverse(poly.begin(), poly.end());
        G4double hzp = (kind == "HOLE") ? hz * 1.02 : hz;
        auto ex = new G4ExtrudedSolid(solidName + "_p" + std::to_string(idx++), poly, hzp,
                                      G4TwoVector(0, 0), 1.0, G4TwoVector(0, 0), 1.0);
        if(kind == "HOLE") holes.push_back(ex);
        else acc = acc ? (G4VSolid *)new G4UnionSolid(solidName + "_u" + std::to_string(idx), acc, ex) : (G4VSolid *)ex;
      }
      for(size_t h = 0; h < holes.size(); ++h)
        acc = new G4SubtractionSolid(solidName + "_s" + std::to_string(h), acc, holes[h]);
      shapeSolid = acc;
    } else {
      G4cerr << "ERROR: unknown POTATO_SHAPE '" << shape << "'." << G4endl;
      exit(EXIT_FAILURE);
    }

    G4cout << "Placing '" << shape << "' [" << objMat->GetName() << "] at (" << cx / mm
           << ", " << cy / mm << ", " << cz / mm << ") R=" << R / mm << " mm ang="
           << ang / deg << " deg" << G4endl;

    G4VSolid *finalSolid = shapeSolid;
    if(shape == "orb" && gapBox) {
      auto clipBox = new G4Box(solidName + "_clip", gapBox->GetXHalfLength(),
                               gapBox->GetYHalfLength(), gapBox->GetZHalfLength());
      finalSolid = new G4IntersectionSolid(solidName, shapeSolid, clipBox, nullptr,
                                           G4ThreeVector(-cx, -cy, -cz));
    }
    auto objLogical = new G4LogicalVolume(finalSolid, objMat, solidName);
    G4VisAttributes vis;
    G4Color blue; G4Color::GetColour("blue", blue); blue.SetAlpha(0.4);
    vis.SetColor(blue); vis.SetForceSolid();
    objLogical->SetVisAttributes(vis);
    G4RotationMatrix *objRot = nullptr;
    if(const char *raxis = envs("ROTAXIS")) {
      G4double rdeg = (envs("ROTDEG") ? std::atof(envs("ROTDEG")) : 90.0) * deg;
      objRot = new G4RotationMatrix();
      std::string ax = raxis;
      if(ax == "x") objRot->rotateX(rdeg);
      else if(ax == "y") objRot->rotateY(rdeg);
      else if(ax == "z") objRot->rotateZ(rdeg);
      else { G4cerr << "ERROR: unknown POTATO_ROTAXIS" << G4endl; exit(EXIT_FAILURE); }
      G4cout << "  object rotation: axis=" << ax << " deg=" << rdeg / deg << G4endl;
    }
    new G4PVPlacement(objRot, {cx, cy, cz}, objLogical, solidName + "_phys", gapLogical, false, 0, true);
  }
  }
}

void DetectorConstruction::DefineFields()
{

  if(G4LogicalVolume *magnetLV = fLogicalVolumeStore->GetVolume("magnet_gap", false)) {
    G4double B = 1.5 * tesla;
    if(const char *bs = getenv("MUPOS_BFIELD")) { B = std::atof(bs) * tesla; }
    auto bfield = new G4UniformMagField(G4ThreeVector{ B, 0.0, 0.0 });
    auto bmanager = new G4FieldManager(bfield);
    bmanager->CreateChordFinder(bfield);
    magnetLV->SetFieldManager(bmanager, true);
    auto blimits = new G4UserLimits(2.0 * mm);
    magnetLV->SetUserLimits(blimits);
    G4AutoDelete::Register(bfield);
    G4AutoDelete::Register(blimits);
    G4cout << "[spectrometer] magnet_gap field B_x = " << B / tesla << " T" << G4endl;
  }

  G4String name = "newrpc_electric";
  std::vector<G4VPhysicalVolume *> newrpc_electrics;
  WalkVolume(NULL, [&name, &newrpc_electrics](G4VPhysicalVolume *v) {
    if(v->GetLogicalVolume()->GetName() == name) { newrpc_electrics.push_back(v); }
  });
  std::sort(newrpc_electrics.begin(), newrpc_electrics.end());
  newrpc_electrics.erase(std::unique(newrpc_electrics.begin(), newrpc_electrics.end()), newrpc_electrics.end());

  G4double z = fScoringHalfZ * 2;
  auto electric = new G4Box("electric", fElectrodeHalfX, fElectrodeHalfY, fScoringHalfZ);

  G4double step = z * 0.01;
  G4double U = 10100 * volt, E = U / z;
  auto field = new G4UniformElectricField(G4ThreeVector{ 0, 0, E });

  auto magField = new G4UniformMagField(G4ThreeVector{ 0, 0, 0 });
  auto manager = new G4FieldManager(field);
  manager->CreateChordFinder(magField);
  G4AutoDelete::Register(field);
  G4AutoDelete::Register(magField);

  for(G4VPhysicalVolume *newrpc_electric : newrpc_electrics) {

    PrintVolumes(newrpc_electric);
    newrpc_electric = PartitionVolume(
        newrpc_electric, [&electric, &name](G4VSolid *solid, const G4ThreeVector &r, const G4RotationMatrix &rm) {

          static size_t g_id;
          size_t id = g_id++;
          std::vector<G4VSolid *> parts;
          parts.reserve(2);
          auto rotation = new G4RotationMatrix(rm.inverse());
          auto rps = -(*rotation * r);
          name = "part_" + std::to_string(id) + "_0_" + solid->GetName();
          parts.push_back(new G4IntersectionSolid(name, solid, electric, rotation, rps));
          name = "part_" + std::to_string(id) + "_1_" + solid->GetName();
          parts.push_back(new G4SubtractionSolid(name, solid, electric, rotation, rps));
          G4AutoDelete::Register(rotation);
          return parts;
        });
    PrintVolumes(newrpc_electric);
    G4VPhysicalVolume *electric_volume = newrpc_electric->GetLogicalVolume()->GetDaughter(0);
    electric_volume->GetLogicalVolume()->SetFieldManager(manager, true);
    {
      G4VisAttributes attr;
      G4Color color;
      G4Color::GetColour("green", color), color.SetAlpha(0.2);
      attr.SetColor(color), attr.SetForceSolid();
      electric_volume->GetLogicalVolume()->SetVisAttributes(attr);
    }

    auto limits = new G4UserLimits(step);
    WalkVolume(electric_volume->GetLogicalVolume(), [limits](G4LogicalVolume *volume) {
      G4cout << "Setting step limit for " << volume->GetName() << G4endl;
      volume->SetUserLimits(limits);
    });
    G4AutoDelete::Register(limits);
  }
}

G4VPhysicalVolume *DetectorConstruction::Construct()
{
  G4GeometryManager::GetInstance()->OpenGeometry();
  G4SolidStore::GetInstance()->Clean();
  fLogicalVolumeStore->Clean();
  fPhysicalVolumeStore->Clean();

  DefineMaterials();
  DefineVolumes();
  DefineFields();
  PrintVolumes(NULL);

  ((PrimaryGeneratorAction *)G4RunManager::GetRunManager()->GetUserPrimaryGeneratorAction())->Initialize(this);
  return fWorld;
}

void DetectorConstruction::PrintVolumes(G4VPhysicalVolume *volume) const
{
  size_t depth = 0;
  WalkVolume(
      volume,
      [&depth](G4VPhysicalVolume *v) {
        G4cout << std::string(2 * depth++, ' ');
        G4cout << v->GetName() << " - " << v->GetLogicalVolume()->GetName() << " - "
               << v->GetLogicalVolume()->GetSolid()->GetName() << G4endl;
      },
      [&depth](G4VPhysicalVolume *) { --depth; });
}

static void WalkVolume(G4LogicalVolume *volume, const std::function<void(G4LogicalVolume *)> &enter,
    const std::function<void(G4LogicalVolume *)> &leave)
{
  if(enter) { enter(volume); }
  for(size_t i = 0; i < volume->GetNoDaughters(); ++i) {
    WalkVolume(volume->GetDaughter(i)->GetLogicalVolume(), enter, leave);
  }
  if(leave) { leave(volume); }
}

void DetectorConstruction::WalkVolume(G4LogicalVolume *volume, const std::function<void(G4LogicalVolume *)> &enter,
    const std::function<void(G4LogicalVolume *)> &leave) const
{
  if(volume == NULL) { volume = fWorld->GetLogicalVolume(); }
  if(volume == NULL) { return; }
  ::WalkVolume(volume, enter, leave);
}

static void WalkVolume(G4VPhysicalVolume *volume, const std::function<void(G4VPhysicalVolume *)> &enter,
    const std::function<void(G4VPhysicalVolume *)> &leave)
{
  if(enter) { enter(volume); }
  G4LogicalVolume *logical = volume->GetLogicalVolume();
  for(size_t i = 0; i < logical->GetNoDaughters(); ++i) { WalkVolume(logical->GetDaughter(i), enter, leave); }
  if(leave) { leave(volume); }
}

void DetectorConstruction::WalkVolume(G4VPhysicalVolume *volume, const std::function<void(G4VPhysicalVolume *)> &enter,
    const std::function<void(G4VPhysicalVolume *)> &leave) const
{
  if(volume == NULL) { volume = fWorld; }
  if(volume == NULL) { return; }
  ::WalkVolume(volume, enter, leave);
}

void DetectorConstruction::WalkVolume(G4VPhysicalVolume *volume,
    const std::function<void(G4VPhysicalVolume *, const G4ThreeVector &, const G4RotationMatrix &)> &enter,
    const std::function<void(G4VPhysicalVolume *, const G4ThreeVector &, const G4RotationMatrix &)> &leave) const
{
  G4ThreeVector r = { 0, 0, 0 };
  G4RotationMatrix rm = { 0, 0, 0 };
  WalkVolume(
      volume,
      [&r, &rm, &enter](G4VPhysicalVolume *v) {
        r += rm * v->GetObjectTranslation();
        if(G4RotationMatrix *rotation = v->GetObjectRotation()) { rm = rm * *rotation; }
        if(enter) { enter(v, r, rm); }
      },
      [&r, &rm, &leave](G4VPhysicalVolume *v) {
        if(leave) { leave(v, r, rm); }
        if(G4RotationMatrix *rotation = v->GetObjectRotation()) { rm = rm * rotation->inverse(); }
        r -= rm * v->GetObjectTranslation();
      });
}

G4VPhysicalVolume *DetectorConstruction::PartitionVolume(G4VPhysicalVolume *volume,
    const std::function<std::vector<G4VSolid *>(G4VSolid *, const G4ThreeVector &, const G4RotationMatrix &)>
        &partition) const
{
  std::vector<std::vector<std::vector<G4LogicalVolume *>>> stack(1);
  WalkVolume(
      volume, [&stack](G4VPhysicalVolume *, const G4ThreeVector &, const G4RotationMatrix &) { stack.emplace_back(); },
      [&stack, &partition](G4VPhysicalVolume *v, const G4ThreeVector &r, const G4RotationMatrix &rm) {
        static size_t g_id;
        size_t id = g_id++;

        auto children = std::move(stack.back());
        stack.pop_back();
        size_t nchild = children.size();

        stack.back().emplace_back();
        auto &result = stack.back().back();

        auto solids = partition(v->GetLogicalVolume()->GetSolid(), r, rm);
        size_t npart = solids.size();
        for(auto &child : children) {
          if(child.size() != npart) { throw std::logic_error("inconsistent number of partitions"); }
        }

        result.reserve(npart);
        for(size_t ipart = 0; ipart < npart; ++ipart) {
          if(solids[ipart] == NULL) {
            for(size_t ichild = 0; ichild < nchild; ++ichild) {
              if(children[ichild][ipart]) { throw std::logic_error("null volume contains non-null child"); }
            }
            result.push_back(NULL);
            continue;
          }

          G4String name =
              "part_" + std::to_string(id) + "_" + std::to_string(ipart) + "_" + v->GetLogicalVolume()->GetName();
          result.push_back(new G4LogicalVolume(solids[ipart], v->GetLogicalVolume()->GetMaterial(), name));
          result.back()->SetVisAttributes(v->GetLogicalVolume()->GetVisAttributes());

          for(size_t ichild = 0; ichild < nchild; ++ichild) {
            if(children[ichild][ipart] == NULL) { continue; }
            G4VPhysicalVolume *dau = v->GetLogicalVolume()->GetDaughter(ichild);
            name = "part_" + std::to_string(id) + "_" + std::to_string(ipart) + "_" + dau->GetName();
            new G4PVPlacement(dau->GetRotation(), dau->GetTranslation(), children[ichild][ipart], name, result.back(),
                false, 0, true);
          }
        }
      });

  static size_t g_id;
  size_t id = g_id++;
  G4String name = "group_" + std::to_string(id) + "_" + volume->GetLogicalVolume()->GetName();
  auto solid = volume->GetLogicalVolume()->GetSolid();
  auto material = volume->GetLogicalVolume()->GetMaterial();
  auto logical = new G4LogicalVolume(solid, material, name);
  logical->SetVisAttributes(volume->GetLogicalVolume()->GetVisAttributes());
  for(size_t ichild = 0; ichild < stack[0][0].size(); ++ichild) {
    if(stack[0][0][ichild] == NULL) { continue; }
    name = "group_" + std::to_string(id) + "_" + std::to_string(ichild) + "_" + volume->GetName();
    new G4PVPlacement(0, { 0, 0, 0 }, stack[0][0][ichild], name, logical, false, 0, true);
  }

  name = "group_" + std::to_string(id) + "_" + volume->GetName();
  auto rotation = volume->GetObjectRotation();
  auto translation = volume->GetObjectTranslation();
  auto mother = volume->GetMotherLogical();
  mother->RemoveDaughter(volume);
  delete volume;
  return new G4PVPlacement(rotation, translation, logical, name, mother, false, 0, true);
}

G4double DetectorConstruction::GetMaxScoringZ() const
{
  return fMaxScoringZ;
}

G4double DetectorConstruction::GetDetectorHalfX() const
{

  return GetScoringHalfX();
}

G4double DetectorConstruction::GetDetectorHalfY() const
{

  return GetScoringHalfY();
}
