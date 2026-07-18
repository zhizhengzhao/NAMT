

#include <Rtypes.h>

#include <tuple>

#ifndef EdepData_h
#define EdepData_h 1

struct EdepKey {
public:
  Int_t Id;
  Int_t Pid;
  Int_t Process;
  Int_t trackID;

  auto Tuple() { return std::tie(Id, Pid, Process, trackID); }
  auto Tuple() const { return std::tie(Id, Pid, Process, trackID); }

};

inline bool operator<(const EdepKey &lhs, const EdepKey &rhs) { return lhs.Tuple() < rhs.Tuple(); }
inline bool operator==(const EdepKey &lhs, const EdepKey &rhs) { return lhs.Tuple() == rhs.Tuple(); }

namespace std {

template<>
struct hash<EdepKey> {
  size_t operator()(const EdepKey &key) const
  {
    return hash<int>()(key.Id) ^ hash<int>()(key.Pid) ^ hash<int>()(key.Process);
  }
};

}

struct EdepValue {
public:
  Double_t Value;
  Double_t X;
  Double_t Y;
  Double_t T;
  Int_t trackID;

  auto Tuple() && { return std::tie(Value, X, Y, T, trackID); }
  auto Tuple() const && { return std::tie(Value, X, Y, T, trackID); }

  EdepValue &Add(Double_t v, Double_t x, Double_t y, Double_t t, Int_t tid)
  {
    if(v > 0 && tid > 0) Value += v, X += v * x, Y += v * y, T += v * t, trackID = tid;
    return *this;
  }

  EdepValue &&Finish()
  {
    if(Value > 0) X /= Value, Y /= Value, T /= Value;
    return std::move(*this);
  }
};

#endif
