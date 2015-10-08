//===-------- MemoryIdempotenceAnalysis.cpp ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation for computing the idempotent region
// information at the LLVM IR level in terms of the "cuts" that define them.
// See "Static Analysis and Compiler Design for Idempotent Processing" in PLDI
// '12.
//
// Potential cut points are captured by the CandidateInfo class, which contains
// some meta-info used in the hitting set computation.
//
//===----------------------------------------------------------------------===//
//
// Some NMC implementations based on Prof. Martin Mueller's nmc.cpp

#define DEBUG_TYPE "memory-idempotence-analysis"
#include "llvm/BasicBlock.h"
#include "llvm/Function.h"
#include "llvm/Instruction.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/CodeGen/IdempotenceOptions.h"
#include "llvm/CodeGen/MemoryIdempotenceAnalysis.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PredIteratorCache.h"
#include <algorithm>
#include <sstream>
#include <vector>
#include <math.h>
//#include <cstdint>
#include <list>
#include <set>

#include <tr1/unordered_set>
#include <tr1/unordered_map>
#include <vector>
#include <tr1/random>
#include <iostream>

using namespace llvm;
using namespace std;
using namespace tr1;

cl::opt<bool> BuildIdempotentRegionsWithGA(
    "idempotence-GA", cl::Hidden,
    cl::desc("Build Idempotent Regions With Genetic Algorithm"),
    cl::init(false));

//NMC Parameter
cl::opt<bool> BuildIdempotentRegionsWithMC(
    "idempotence-MC", cl::Hidden,
    cl::desc("Build Idempotent Regions With NMC Algorithm"),
    cl::init(false));
//NMC Parameter

cl::opt<int> POPULATION_SIZE(
    "idempotence-population-size", cl::Hidden,
    cl::desc("Population Size for the Genetic Algorithm of the Idempotence Analysis"),
    cl::init(100));
//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

//NMC Classes

bool cmp(std::pair<uint64_t,int> const & a, std::pair<uint64_t,int> const & b)
{
    return a.second > b.second;
}

class AMAFStats{
    
public:
    int bestScore;
    int numVisit;
    int totalScore;

    AMAFStats():bestScore(std::numeric_limits<int>::max()), numVisit(0), totalScore(0){}
    
};

class HSState{

public:
    HSState():score(0){}
    ~HSState(){}
    
    HSState(const HSState& mHSState){
        elementCount = mHSState.elementCount;
        setList = mHSState.setList;
        elementArray = mHSState.elementArray;
        score = mHSState.score;
    }
    
    list<set<uint64_t> > setList;
    unordered_map<uint64_t,int> elementArray;
    int elementCount;
    int score;
    
    void InitSetList(list<set<uint64_t> >& _setList){
        elementCount = 0;
        setList = _setList;
        for (list<set<uint64_t> >::iterator listIter = _setList.begin(); listIter != _setList.end(); ++listIter){
            set<uint64_t> mSet = *listIter;
            for (set<uint64_t>::iterator setIter = mSet.begin(); setIter != mSet.end(); ++setIter){
                uint64_t element = *setIter;
                unordered_map<uint64_t, int>::const_iterator got = elementArray.find(element);
                if ( got == elementArray.end() ){
                    elementArray.insert(make_pair(element,1));
                    ++elementCount;
                }else{//else find
                    elementArray[element]++;
                }
            }
        }
    }
    
    void Hit(uint64_t element){
         for(list<set<uint64_t> >::iterator listIter = setList.begin(); listIter != setList.end();)
        {
            std::set<uint64_t>::iterator got = listIter->find(element);
            if(got != listIter->end()){//hit
                set<uint64_t> mSet = *listIter;
                for (set<uint64_t>::iterator elementIter = mSet.begin(); elementIter != mSet.end(); ++elementIter){
                    elementArray[*elementIter]--;
                    if (elementArray[*elementIter] == 0){
                        elementArray.erase(*elementIter);
                    }
                }
                listIter = setList.erase(listIter);
            }
            else{
                ++listIter;
            }
        }
        elementArray.erase(element);
        ++score;
    }
    
    bool IsAllHit(){
        return setList.empty();
    }

};

//NMC Classes

static bool isSubloopPreheader(const BasicBlock &BB,
                               const LoopInfo &LI) {
  Loop *L = LI.getLoopFor(&BB);
  if (L)
    for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
      if (&BB == (*I)->getLoopPreheader())
        return true;
  return false;
}

static std::string getLocator(const Instruction &I) {
  unsigned Offset = 0;
  const BasicBlock *BB = I.getParent();
  for (BasicBlock::const_iterator It = I; It != BB->begin(); --It)
    ++Offset;

  std::stringstream SS;
  SS << BB->getName().str() << ":" << Offset;
  return SS.str();
}

namespace {
  typedef std::pair<Instruction *, Instruction *> AntidependencePairTy;
  typedef SmallVector<Instruction *, 16> AntidependencePathTy;
}

namespace llvm {
  static raw_ostream &operator<<(raw_ostream &OS, const AntidependencePairTy &P);
  static raw_ostream &operator<<(raw_ostream &OS, const AntidependencePathTy &P);
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const AntidependencePairTy &P) {
  OS << "Antidependence Pair (" << getLocator(*P.first) << ", " 
    << getLocator(*P.second) << ")";
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const AntidependencePathTy &P) {
  OS << "[";
  for (AntidependencePathTy::const_iterator I = P.begin(), First = I,
       E = P.end(); I != E; ++I) {
    if (I != First)
      OS << ", ";
    OS << getLocator(**I);
  }
  OS << "]";
  return OS;
}

//===----------------------------------------------------------------------===//
// CandidateInfo
//===----------------------------------------------------------------------===//

namespace {
  class CandidateInfo {
   public:
    typedef SmallPtrSet<const AntidependencePathTy *, 4> UnintersectedPaths;

    // Constructor.
    CandidateInfo(Instruction *Candidate,
                  unsigned LoopDepth,
                  bool IsSubloopPreheader);

    // Get the candidate instruction.
    Instruction *getCandidate() { return Candidate_; }
    const Instruction *getCandidate() const { return Candidate_; }

    // Iteration support (const only).
    typedef UnintersectedPaths::const_iterator const_iterator;
    const_iterator begin()  const { return UnintersectedPaths_.begin(); }
    const_iterator end()    const { return UnintersectedPaths_.end(); }
    unsigned       size()   const { return UnintersectedPaths_.size(); }
    bool           empty()  const { return UnintersectedPaths_.empty(); }

    // Add Path to the set of unintersected paths and update priority.
    void add(const AntidependencePathTy &Path);

    // Remove Path from the set of unintersected paths and update priority.
    void remove(const AntidependencePathTy &Path);

    // Debugging support.
    void print(raw_ostream &OS) const;

    // Priority comparison function.
    static bool compare(CandidateInfo *L, CandidateInfo *R) {
      return (L->Priority_ < R->Priority_);
    }

   private:
    Instruction *Candidate_;
    UnintersectedPaths UnintersectedPaths_;

    union {
      // Higher priority is better.
      struct {
        // From least important to most important (little endian):
        signed IntersectedPaths:16;    // prefer more already-intersected paths
        signed IsSubloopPreheader:8;   // prefer preheaders
        signed IsAntidependentStore:8; // prefer antidependent stores
        signed UnintersectedPaths:16;  // prefer more unintersected paths
        signed LoopDepth:16;           // (inverted) prefer outer loops
      } PriorityElements_;
      uint64_t Priority_;
    };

    // Do not implement.
    CandidateInfo();
  };

  typedef std::vector<CandidateInfo *> WorklistTy; 
} // end anonymous namespace

CandidateInfo::CandidateInfo(Instruction *Candidate,
                             unsigned LoopDepth,
                             bool IsSubloopPreheader)
    : Candidate_(Candidate), Priority_(0) {
  PriorityElements_.LoopDepth = ~LoopDepth;
  PriorityElements_.IsAntidependentStore = false;
  PriorityElements_.IsSubloopPreheader = IsSubloopPreheader;
  PriorityElements_.UnintersectedPaths = 0;
  PriorityElements_.IntersectedPaths = 0;
}

void CandidateInfo::print(raw_ostream &OS) const {
  OS << "Candidate " << getLocator(*Candidate_)
    << "\n Priority:              " << Priority_
    << "\n  LoopDepth:            " << PriorityElements_.LoopDepth
    << "\n  UnintersectedPaths:   " << PriorityElements_.UnintersectedPaths
    << "\n  IsAntidependentStore: " << PriorityElements_.IsAntidependentStore
    << "\n  IsSubloopPreheader:   " << PriorityElements_.IsSubloopPreheader
    << "\n  IntersectedPaths:     " << PriorityElements_.IntersectedPaths
    << "\n";
}

void CandidateInfo::add(const AntidependencePathTy &Path) {
  // Antidependent stores are always the first store on the path.
  if (Candidate_ == *Path.begin())
    PriorityElements_.IsAntidependentStore = true;

  // Update other structures.
  PriorityElements_.UnintersectedPaths++;
  assert(UnintersectedPaths_.insert(&Path) && "already inserted");
}

void CandidateInfo::remove(const AntidependencePathTy &Path) {
  // Update priority.
  PriorityElements_.UnintersectedPaths--;
  PriorityElements_.IntersectedPaths++;
  assert(PriorityElements_.UnintersectedPaths >= 0 &&
         PriorityElements_.IntersectedPaths >= 0 && "Wrap around");

  // Remove Path from the list of unintersected paths.
  assert(UnintersectedPaths_.erase(&Path) && "path not in set");
  assert(static_cast<unsigned>(PriorityElements_.UnintersectedPaths) ==
         UnintersectedPaths_.size());
}

//===----------------------------------------------------------------------===//
// MemoryIdempotenceAnalysisImpl
//===----------------------------------------------------------------------===//

typedef SmallVector<AntidependencePathTy, 16> AntidependencePaths;
typedef SmallVector<AntidependencePairTy, 16> AntidependencePairs;

class llvm::MemoryIdempotenceAnalysisImpl {
 private:
  // Constructor.
  MemoryIdempotenceAnalysisImpl(MemoryIdempotenceAnalysis *MIA) : MIA_(MIA) {}

  // Forwarded function implementations.
  void releaseMemory();
  void print(raw_ostream &OS, const Module *M = 0) const;
  bool runOnFunction(Function &F);

 private:
  friend class MemoryIdempotenceAnalysis;
  MemoryIdempotenceAnalysis *MIA_;

  // Final output structure.
  MemoryIdempotenceAnalysis::CutSet CutSet_;

  // Intermediary data structure 1.
  AntidependencePairs AntidependencePairs_;

  // Intermediary data structure 2.
  AntidependencePaths AntidependencePaths_;

  // Other things we use.
  PredIteratorCache PredCache_;
  Function *F_;
  AliasAnalysis *AA_;
  DominatorTree *DT_;
  LoopInfo *LI_;

  // Helper functions.
  void forceCut(BasicBlock::iterator I);
  void findAntidependencePairs(StoreInst *Store);
  bool scanForAliasingLoad(BasicBlock::iterator I,
                           BasicBlock::iterator E,
                           StoreInst *Store,
                           Value *Pointer,
                           unsigned PointerSize);
  void computeAntidependencePaths();
  void computeHittingSet();
  void computeCutSetWithGA();
  int nmc(int level, int tries, HSState& initState);
  void computeCutSetWithMC();
  std::list< std::set<uint64_t> > getSets();

  void processRedundantCandidate(CandidateInfo *RedundantInfo,
                                 WorklistTy *Worklist,
                                 const AntidependencePathTy &Path);
};

std::list< std::set<uint64_t> > MemoryIdempotenceAnalysisImpl::getSets(){
	list< set<uint64_t> > result;
	for(AntidependencePaths::iterator P_it = AntidependencePaths_.begin(), P_end = AntidependencePaths_.end(); P_it != P_end; P_it++){

		set<uint64_t> currentSet;

		for(AntidependencePathTy::iterator Iit = P_it->begin(), Iend = P_it->end(); Iit != Iend; Iit++){

			Instruction* I = *Iit;
			currentSet.insert((uint64_t)I);

		}

		result.push_back(currentSet);

	}

	return result;
}


void MemoryIdempotenceAnalysisImpl::releaseMemory() {
  CutSet_.clear();
  AntidependencePairs_.clear();
  AntidependencePaths_.clear();
  PredCache_.clear();
}

static bool forcesCut(const Instruction &I) {
  // See comment at the head of forceCut() further below.
  if (const LoadInst *L = dyn_cast<LoadInst>(&I))
    return L->isVolatile();
  if (const StoreInst *S = dyn_cast<StoreInst>(&I))
    return S->isVolatile();
  return (isa<CallInst>(I) ||
          isa<InvokeInst>(I) ||
          isa<VAArgInst>(&I) ||
          isa<FenceInst>(&I) ||
          isa<AtomicCmpXchgInst>(&I) ||
          isa<AtomicRMWInst>(&I));
}

bool MemoryIdempotenceAnalysisImpl::runOnFunction(Function &F) {
  F_  = &F;
  AA_ = &MIA_->getAnalysis<AliasAnalysis>();
  DT_ = &MIA_->getAnalysis<DominatorTree>();
  LI_ = &MIA_->getAnalysis<LoopInfo>();
  DEBUG(dbgs() << "\n*** MemoryIdempotenceAnalysis for Function "
        << F_->getName() << " ***\n");

  DEBUG(dbgs() << "\n** Computing Forced Cuts\n");
  int numInsts = 0;

  for (Function::iterator BB = F.begin(); BB != F.end(); ++BB)
    for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ++I) {
    	numInsts++;
    	if (forcesCut(*I)) forceCut(I);
    }

  DEBUG(dbgs() << "\n** Computing Memory Antidependence Pairs\n");
  for (Function::iterator BB = F.begin(); BB != F.end(); ++BB)
    for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ++I)
      if (StoreInst *Store = dyn_cast<StoreInst>(I))
        findAntidependencePairs(Store);

  // Return early if there's nothing to analyze.
  if (AntidependencePairs_.empty())
    return false;

  DEBUG(dbgs() << "\n** Computing Paths to Cut\n");
  computeAntidependencePaths();

  if(!BuildIdempotentRegionsWithGA && !BuildIdempotentRegionsWithMC){//add NMC parameter
	  DEBUG(dbgs() << "\n** Computing Hitting Set\n");
	  computeHittingSet();
  } else if (BuildIdempotentRegionsWithGA){

	  if (POPULATION_SIZE > numInsts){
		  if (numInsts < 10) POPULATION_SIZE = 10;
		  else POPULATION_SIZE = numInsts;
	  }

	  //Use a Genetic Algorithm to create the set
	  computeCutSetWithGA();

  }else{//compute cut set with MNC
	  computeCutSetWithMC();
  }

  DEBUG(print(dbgs()));
  return false;
}

void MemoryIdempotenceAnalysisImpl::forceCut(BasicBlock::iterator I) {
  // These cuts actually need to occur at the machine level.  Calls and invokes
  // are one common case that we are handled after instruction selection; see
  // patchCallingConvention() in PatchMachineIdempotentRegions.  In the absence
  // of any actual hardware support, the others are just approximated here.
  if (CallSite(I))
    return;

  DEBUG(dbgs() << " Inserting forced cut at " << getLocator(*I) << "\n");
  CutSet_.insert(++I);
}

void MemoryIdempotenceAnalysisImpl::findAntidependencePairs(StoreInst *Store) {
  DEBUG(dbgs() << " Analyzing store " << getLocator(*Store) << "\n");
  Value *Pointer = Store->getOperand(1);
  unsigned PointerSize = AA_->getTypeStoreSize(Store->getOperand(0)->getType());

  // Perform a reverse depth-first search to find aliasing loads.
  typedef std::pair<BasicBlock *, BasicBlock::iterator> WorkItem;
  SmallVector<WorkItem, 8> Worklist;
  SmallPtrSet<BasicBlock *, 32> Visited;

  BasicBlock *StoreBB = Store->getParent();
  Worklist.push_back(WorkItem(StoreBB, Store));
  do {
    BasicBlock *BB;
    BasicBlock::iterator I, E;
    tie(BB, I) = Worklist.pop_back_val();

    // If we are revisiting StoreBB, we scan to Store to complete the cycle.
    // Otherwise we end at BB->begin().
    E = (BB == StoreBB && I == BB->end()) ? Store : BB->begin();

    // Scan for an aliasing load.  Terminate this path if we see one or a cut is
    // already forced.
    if (scanForAliasingLoad(I, E, Store, Pointer, PointerSize))
      continue;

    // If the path didn't terminate, continue on to predecessors.
    for (BasicBlock **P = PredCache_.GetPreds(BB); *P; ++P)
      if (Visited.insert(*P))
        Worklist.push_back(WorkItem((*P), (*P)->end()));

  } while (!Worklist.empty());
}

bool MemoryIdempotenceAnalysisImpl::scanForAliasingLoad(BasicBlock::iterator I,
                                                        BasicBlock::iterator E,
                                                        StoreInst *Store,
                                                        Value *Pointer,
                                                        unsigned PointerSize) {
  while (I != E) {
    --I;
    // If we see a forced cut, the path is already cut; don't scan any further.
    if (forcesCut(*I))
      return true;

    // Otherwise, check for an aliasing load.
    if (LoadInst *Load = dyn_cast<LoadInst>(I)) {
      if (AA_->getModRefInfo(Load, Pointer, PointerSize) & AliasAnalysis::Ref) {
        AntidependencePairTy Pair = AntidependencePairTy(I, Store);
        DEBUG(dbgs() << "  " << Pair << "\n");
        AntidependencePairs_.push_back(Pair);
        return true;
      }
    }
  }
  return false;
}

void MemoryIdempotenceAnalysisImpl::computeAntidependencePaths() {

  // Compute an antidependence path for each antidependence pair.
  for (AntidependencePairs::iterator I = AntidependencePairs_.begin(), 
       E = AntidependencePairs_.end(); I != E; ++I) {
    BasicBlock::iterator Load, Store;
    tie(Load, Store) = *I;

    // Prepare a new antidependence path.
    AntidependencePaths_.resize(AntidependencePaths_.size() + 1);
    AntidependencePathTy &Path = AntidependencePaths_.back();

    // The antidependent store is always on the path.
    Path.push_back(Store);

    // The rest of the path consists of other stores that dominate Store but do
    // not dominate Load.  Handle the block-local case quickly.
    BasicBlock::iterator Cursor = Store;
    BasicBlock *SBB = Store->getParent(), *LBB = Load->getParent();
    if (SBB == LBB && DT_->dominates(Load, Store)) {
      while (--Cursor != Load)
        if (isa<StoreInst>(Cursor))
          Path.push_back(Cursor);
      DEBUG(dbgs() << " Local " << *I << " has path " << Path << "\n");
      continue;
    }

    // Non-local case.
    BasicBlock *BB = SBB;
    DomTreeNode *DTNode = DT_->getNode(BB), *LDTNode = DT_->getNode(LBB);
    while (!DT_->dominates(DTNode, LDTNode)) {
      DEBUG(dbgs() << "  Scanning dominating block " << BB->getName() << "\n");
      BasicBlock::iterator E = BB->begin();
      while (Cursor != E)
        if (isa<StoreInst>(--Cursor))
          Path.push_back(Cursor);

      // Move the cursor to the end of BB's IDom block.
      DTNode = DTNode->getIDom();
      if (DTNode == NULL)
        break;
      BB = DTNode->getBlock();
      Cursor = BB->end();
    }
    DEBUG(dbgs() << " Non-local " << *I << " has path " << Path << "\n");
  }
}

static void dumpWorklist(const WorklistTy &Worklist) {
  dbgs() << "Worklist:\n";
  for (WorklistTy::const_iterator I = Worklist.begin(), E = Worklist.end();
       I != E; ++I)
    (*I)->print(dbgs());
  dbgs() << "\n";
}

void MemoryIdempotenceAnalysisImpl::computeHittingSet() {
  // This function does not use the linear-time version of the hitting set
  // approximation algorithm, which requires constant-time lookup and
  // constant-time insertion data structures.  This doesn't mesh well with
  // a complex priority function such as ours.  This implementation adds a
  // logarithmic factor using a sorted worklist to track priorities.  Although
  // the time complexity is slightly higher, it is much more space efficient as
  // a result.
  typedef DenseMap<const Instruction *, CandidateInfo *> CandidateInfoMapTy;
  CandidateInfoMapTy CandidateInfoMap;

  // Find all candidates and compute their priority.
  for (AntidependencePaths::iterator I = AntidependencePaths_.begin(),
       IE = AntidependencePaths_.end(); I != IE; ++I) {
    AntidependencePathTy &Path = *I;
    for (AntidependencePathTy::iterator J = Path.begin(), JE = Path.end();
         J != JE; ++J) {
      Instruction *Candidate = *J;
      BasicBlock *CandidateBB = Candidate->getParent();
      CandidateInfo *&CI = CandidateInfoMap[Candidate];
      if (CI == NULL)
        CI = new CandidateInfo(Candidate,
                               LI_->getLoopDepth(CandidateBB),
                               isSubloopPreheader(*CandidateBB, *LI_));
      CI->add(Path);
    }
  }

  // Set up a worklist sorted by priority.  The highest priority candidates
  // will be at the back of the list.
  WorklistTy Worklist; 
  for (CandidateInfoMapTy::iterator I = CandidateInfoMap.begin(),
       E = CandidateInfoMap.end(); I != E; ++I)
    Worklist.push_back(I->second);
  std::sort(Worklist.begin(), Worklist.end(), CandidateInfo::compare);
  DEBUG(dumpWorklist(Worklist));

  // Process the candidates in the order that we see them popping off the back
  // of the worklist.
  while (!Worklist.empty()) {
    CandidateInfo *Info = Worklist.back();
    Worklist.pop_back();

    // Skip over candidates with no unintersected paths.
    if (Info->size() == 0)
      continue;

    // Pick this candidate and put it in the hitting set.
    DEBUG(dbgs() << "Picking "; Info->print(dbgs()));
    CutSet_.insert(Info->getCandidate());

    // For each path that the candidate intersects, the other candidates that
    // also intersect that path now intersect one fewer unintersected paths.
    // Update those candidates (changes their priority) and intelligently
    // re-insert them into the worklist at the right place.
    for (CandidateInfo::const_iterator I = Info->begin(), IE = Info->end();
         I != IE; ++I) {
      DEBUG(dbgs() << " Processing redundant candidates for " << **I << "\n");
      for (AntidependencePathTy::const_iterator J = (*I)->begin(),
           JE = (*I)->end(); J != JE; ++J)
        if (*J != Info->getCandidate())
          processRedundantCandidate(CandidateInfoMap[*J], &Worklist, **I);
    }
  }

  // Clean up.
  for (CandidateInfoMapTy::iterator I = CandidateInfoMap.begin(),
       E = CandidateInfoMap.end(); I != E; ++I)
    delete I->second;
}

//NMC Implementation

cl::opt<int> NUMBER_OF_RUNS(
    "idempotence-mc-num-runs", cl::Hidden,
    cl::desc("Number of runs for the Nested Monte Carlo Tree Search of the Idempotence Analysis"),
    cl::init(10));

cl::opt<int> NESTED_LEVEL(
    "idempotence-mc-nested-level", cl::Hidden,
    cl::desc("Number of level nested for the Nested Monte Carlo Tree Search of the Idempotence Analysis"),
    cl::init(1));

cl::opt<int> MOVE_CHOOSE_POLICY(
    "idempotence-mc-move-choose-policy", cl::Hidden,
    cl::desc("Determine how to choose a child move for starting sampling, 1: All moves, 2: No_AMAF – sample the move without any AMAF stats, 3: No_AMAF_Biased – sample the move can hit the most set currently and without AMAF stats first."),
    cl::init(1));

cl::opt<int> TIEBREAK_POLICY(
    "idempotence-mc-tiebreak-policy", cl::Hidden,
    cl::desc("The tiebreak policy if the best AMAF score is the same. 1: AVG, 2: Frequency, 3:Hitting, 4.Random"),
    cl::init(1));

typedef uint64_t Move;

unsigned long long rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((unsigned long long)hi << 32) | lo;
}

template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
    uniform_int<> dis(0, std::distance(start, end) - 1);
    advance(start, dis(g));
    return start;
}

HSState play(HSState p, Move element)
{
    assert(p.elementCount > 0);
    HSState after = HSState(p);
    after.Hit(element);
    return after;
}

int sample(HSState p, list<Move>& playedMoves)
{

    static random_device rd;
    static mt19937 gen(rdtsc());
//    gen.seed(rdtsc());

    while (!p.IsAllHit()) {
        Move element = (*select_randomly(p.elementArray.begin(), p.elementArray.end(), gen)).first;
        p.Hit(element);
        playedMoves.push_back(element);
    }
    return p.score;
}

int nested (HSState p, int level, list<Move>& Moves)
{
    list<Move> playedMoves;
    
    unordered_map<Move, AMAFStats> AMAFtable;
    for (unordered_map<Move,int>::iterator it = p.elementArray.begin(); it != p.elementArray.end(); ++it){
        Move key = it->first;
        AMAFtable.insert(make_pair(key, AMAFStats()));
    }
    
    int bestGlobalScore = std::numeric_limits<int>::max();
    if (p.IsAllHit())
        bestGlobalScore = p.score;
    
    while (!p.IsAllHit())
    {
        int bestScore = std::numeric_limits<int>::max();
        list<Move> bestMoves;

	std::vector<std::pair<Move,int> > items(p.elementArray.size());
        copy(p.elementArray.begin(), p.elementArray.end(), items.begin());
	if (MOVE_CHOOSE_POLICY == 3){//if biased, we sort all moves by priority
            std::sort(items.begin(), items.end(), cmp);
	}

        //for (unordered_set<Move>::iterator it = p.elementArray.begin(); it != p.elementArray.end(); ++it) // try move
        //for (unordered_map<Move, AMAFStats>::iterator it = AMAFtable.begin(); it != AMAFtable.end(); ++it) // try move
        for (std::vector<pair<Move,int> >::iterator it = items.begin(); it !=items.end(); ++it) // try move
        {
	    Move move = it->first;
            if (MOVE_CHOOSE_POLICY != 1 && AMAFtable[move].numVisit != 0) {
                continue;
            }

            int score;
            list<Move> sampledMove = playedMoves;
            if (level == 1){
                score = sample(play(p, move), sampledMove);
            }
            else{
                score = nested(play(p, move), level - 1, sampledMove);
            }
            
            sampledMove.push_front(move);
            if (score < bestScore)
            {
                bestScore = score;                
                bestMoves = sampledMove;
            }
            
            for (list<Move>::iterator it = sampledMove.begin(); it != sampledMove.end(); ++it){
		Move move = *it;
                if (score < AMAFtable[move].bestScore){
                    AMAFtable[move].bestScore = score;
                }
                AMAFtable[move].numVisit++;
                AMAFtable[move].totalScore += score;
            }
        }
        
        AMAFStats bestMoveStats = AMAFtable[p.elementArray.begin()->first];
        Move bestMove = p.elementArray.begin()->first;
        for (unordered_map<Move,int>::iterator it = p.elementArray.begin(); it != p.elementArray.end(); ++it){
            AMAFStats moveStats = AMAFtable[it->first];
            if (moveStats.bestScore < bestMoveStats.bestScore){
                bestMove = it->first;
                bestMoveStats = moveStats;
            }else if (moveStats.bestScore == bestMoveStats.bestScore){//tie breaker
		switch (TIEBREAK_POLICY) {
		case 1: //Avg
                    if ((float)(moveStats.totalScore/moveStats.numVisit) < (float)(bestMoveStats.totalScore/bestMoveStats.numVisit)){
                    	bestMove = it->first;
                    	bestMoveStats = moveStats;
                    }
		    break;
		case 2: //Frequency
                    if (moveStats.numVisit > bestMoveStats.numVisit){
                    	bestMove = it->first;
                    	bestMoveStats = moveStats;
                    }
		    break;
		case 3: //Hitting
                    if (moveStats.numVisit > bestMoveStats.numVisit){
                    	bestMove = it->first;
                    	bestMoveStats = moveStats;
                    }
		    break;
		case 4: //Random
		    srand(rdtsc());
		    int r = rand();
		    if ((r % 2) == 0){
                    	bestMove = it->first;
                    	bestMoveStats = moveStats;
                    }
		    break;
		}
            }
        }

        if (bestScore < bestGlobalScore)
        {
            bestGlobalScore = bestScore;
            Moves = bestMoves;
        }
        p = play (p, bestMove);
        playedMoves.push_back(bestMove);
    }
//    assert(bestGlobalScore < std::numeric_limits<int>::max());
    return bestGlobalScore;
}

int MemoryIdempotenceAnalysisImpl::nmc(int level, int tries, HSState& initState)
{
    
    int best = std::numeric_limits<int>::max();
    list<Move> bestMoves;
    for (int n = 0; n < tries; ++n)
    {
        HSState p = HSState(initState);
        list<Move> playedMoves;
        int count = nested(p, level, playedMoves);
	if (count < best){
	    best = count;
	    bestMoves = playedMoves;
	}
    }

    for (list<Move>::iterator it = bestMoves.begin(); it != bestMoves.end(); ++it){
	    Move element = *it;
	    CutSet_.insert((Instruction *)element);
    }
    return 0;
}

void MemoryIdempotenceAnalysisImpl::computeCutSetWithMC(){
	list<set<Move> > mSetList = getSets();
	HSState newState;
        newState.InitSetList(mSetList);
        nmc(NESTED_LEVEL, NUMBER_OF_RUNS, newState);
}

//NMC Implementation

////////////////////////////////////////////////////////////////////////////
//		GA IMPLEMENTATION
////////////////////////////////////////////////////////////////////////////

//Parameters for the Genetic Algorithm
cl::opt<int> NUM_GENERATIONS(
    "idempotence-num-generations", cl::Hidden,
    cl::desc("Number of Generations for the Genetic Algorithm of the Idempotence Analysis"),
    cl::init(100));



cl::opt<int> TOURNAMENT_SIZE(
    "idempotence-tournament-size", cl::Hidden,
    cl::desc("Tournament Size for the Genetic Algorithm of the Idempotence Analysis"),
    cl::init(2));

cl::opt<double> CROSSOVER_PROBABILITY(
    "idempotence-crossover-probability", cl::Hidden,
    cl::desc("Crossover Probability for the Genetic Algorithm of the Idempotence Analysis"),
    cl::init(0.8));

cl::opt<double> MUTATION_PROBABILITY(
    "idempotence-mutation-probability", cl::Hidden,
    cl::desc("Mutation Probability for the Genetic Algorithm of the Idempotence Analysis"),
    cl::init(0.1));

cl::opt<bool> USE_ELITISM(
    "idempotence-use-elitism", cl::Hidden,
    cl::desc("Elitism for the Genetic Algorithm of the Idempotence Analysis"),
    cl::init(true));

cl::opt<int> RAND_SEED(
    "idempotence-rand-seed", cl::Hidden,
    cl::desc("Random Number Generator Seed for the GA of the Idempotence Analysis"),
    cl::init(0));

cl::opt<bool> PRINT_STATISTICS(
    "idempotence-print-statistics", cl::Hidden,
    cl::desc("Elitism for the Genetic Algorithm of the Idempotence Analysis"),
    cl::init(false));


class InstCut {
public:
	Instruction* inst;
	bool isCut;
	InstCut(Instruction* p_inst, bool p_isCut): inst(p_inst), isCut(p_isCut) {};
};

class Individual {
	public:
	std::vector<InstCut> instructions;
	int Fitness;

	void updateFitness(){
		Fitness = 0;

		for(unsigned int i = 0; i < instructions.size(); i++){
			if (instructions[i].isCut) Fitness++;
		}

	}

	//Create random individual
	Individual(Function* F){

		int cutProbability = rand() % 1000;

		for(Function::iterator BBit = F->begin(), BBend = F->end(); BBit != BBend; BBit++){
			for(BasicBlock::iterator Iit = BBit->begin(), Iend = BBit->end(); Iit != Iend; Iit++){
				instructions.push_back(InstCut(Iit, cutProbability < rand() % 1000));
			}
		}

		updateFitness();
	};

	~Individual(){
		instructions.clear();
	};

    static bool compare(Individual L, Individual R) {
      return (L.Fitness < R.Fitness);
    }

};

typedef std::vector<Individual> PopulationTy;
PopulationTy Population;
PopulationTy NextPopulation;


void generateRandomPopulation(Function* F, PopulationTy &p_Population){

	p_Population.clear();
	for(int i = 0; i < POPULATION_SIZE; i++){
		p_Population.push_back(Individual(F));
	}

}

void updatePopulationFitness(PopulationTy &p_Population){

	for(unsigned int i = 0; i < p_Population.size(); i++){
		p_Population[i].updateFitness();
	}

}

Individual tournament(PopulationTy &p_Population){

	PopulationTy tournamentMembers;

	for (int i = 0; i < TOURNAMENT_SIZE; i++){
		Individual Novo(p_Population[rand() % p_Population.size()]);
		tournamentMembers.push_back(Novo);
	}

	//Sort the members ascending by the fitness. The best element will be the first member.
	std::sort(tournamentMembers.begin(), tournamentMembers.end(), Individual::compare);
	return Individual(tournamentMembers[0]);

}

void doMutation(Individual &Ind){

	int changeProbability = rand() % 1000;

	for(unsigned int i = 0; i < Ind.instructions.size(); i++){
		if((rand() % 1000) < changeProbability){
			Ind.instructions[i].isCut = !Ind.instructions[i].isCut;
		}
	}

}

void doCrossOver(Individual &Ind1, Individual &Ind2){

	int changePoint = rand() % Ind1.instructions.size();

	for(unsigned int i = changePoint; i < Ind1.instructions.size(); i++){

		bool temp = Ind1.instructions[i].isCut;
		Ind1.instructions[i].isCut = Ind2.instructions[i].isCut;
		Ind2.instructions[i].isCut = temp;

	}

}

void applyGeneticOperators(PopulationTy &p_Population){

	for(unsigned int i = 0; i < p_Population.size(); i+=2){

		if(i < p_Population.size()-1){

			if ((rand() % 1000000) < (1000000 * CROSSOVER_PROBABILITY))
				doCrossOver(p_Population[i], p_Population[i+1]);
		}

	}

	for(unsigned int i = 0; i < p_Population.size(); i++){

		if ((rand() % 1000000) < (1000000 * MUTATION_PROBABILITY)) doMutation(p_Population[i]);

	}

}

void replacePopulation(PopulationTy &oldPopulation, PopulationTy &newPopulation){

	std::sort(oldPopulation.begin(), oldPopulation.end(), Individual::compare);
	std::sort(newPopulation.begin(), newPopulation.end(), Individual::compare);
	Individual x(oldPopulation[0]);

	oldPopulation.clear();

	while(newPopulation.size() > 0){

		if(newPopulation.size() == 1 && USE_ELITISM) {

			//Put the best element of the last generation in the place of the worst element of the new generation
			oldPopulation.push_back(x);
			newPopulation.pop_back();

		} else {

			Individual clone(newPopulation.back());
			oldPopulation.push_back(clone);
			newPopulation.pop_back();

		}

	}

}


void fixInvalidIndividual(AntidependencePaths Paths, std::map<Instruction*, bool> PathMap, Individual &Ind){

	std::map<Instruction*, int> InstructionMap;

	for(unsigned int i = 0; i < Ind.instructions.size(); i++){

		//Remove cuts in PHI nodes
		if (isa<PHINode>(Ind.instructions[i].inst)) {
			Ind.instructions[i].isCut = false;
		}

		InstructionMap[Ind.instructions[i].inst] = i;
	}


	bool hasCut;

	//verify if there is at least a cut in all the Paths
	for(unsigned int i = 0; i < Paths.size(); i++){

		hasCut = false;

		for(unsigned int j = 0; j < Paths[i].size(); j++){

			//One instruction being a cut is enough for each path
			if (Ind.instructions[InstructionMap[Paths[i][j]]].isCut) {
				hasCut = true;
				break;
			}

		}

		if (!hasCut) {
			unsigned int cutPlace = rand() % Paths[i].size();

			while (isa<PHINode>(Paths[i][cutPlace])) cutPlace++;

			cutPlace = cutPlace % Paths[i].size();

			if (isa<PHINode>(Ind.instructions[InstructionMap[Paths[i][cutPlace]]].inst)) errs() << "Deu Merda (Inserindo corte em instrução PHI)\n";

			Ind.instructions[InstructionMap[Paths[i][cutPlace]]].isCut = true;
		}
	}


	for(unsigned int i = 0; i < Ind.instructions.size(); i++){

		//Remove unnecessary cuts
		if (Ind.instructions[i].isCut && !PathMap[Ind.instructions[i].inst]) {
			Ind.instructions[i].isCut = false;
		}

	}

	Ind.updateFitness();

	if(Ind.Fitness == 0 && Paths.size() > 0){
		errs() << "Deu Merda\n";
	}

}


void fixInvalidIndividuals(AntidependencePaths Paths, std::map<Instruction*, bool> PathMap, PopulationTy &p_Population){

	for(unsigned int i = 0; i < p_Population.size(); i++){

		fixInvalidIndividual(Paths, PathMap, p_Population[i]);

	}

}

void printPopulation(PopulationTy &Pop){


	for(int i = 0; i < POPULATION_SIZE; i++){

		errs() << "Fitness: " << Pop[i].Fitness <<"	Dump:";

		for(unsigned int j = 0; j < Pop[i].instructions.size(); j++){
			if (Pop[i].instructions[j].isCut) errs() << "1";
			else errs() << "0";
		}
		errs() << '\n';
	}

}

void printGenerationStatisticsHeader(){
	errs() << "Geração	Mínimo	Média	Máximo	Desvio Padrão\n";
}


void printGenerationStatistics(int generation, PopulationTy &Pop){

	if(generation == 0) printGenerationStatisticsHeader();

	double Min = Pop[0].Fitness;
	double Max = Pop[0].Fitness;
	double Avg = Pop[0].Fitness;

	for(int i = 1; i < POPULATION_SIZE; i++){

		if (Pop[i].Fitness > Max) Max = Pop[i].Fitness;
		if (Pop[i].Fitness < Min) Min = Pop[i].Fitness;

		Avg += Pop[i].Fitness;

	}

	Avg = Avg / POPULATION_SIZE;

	double Acum = 0;

	for(int i = 0; i < POPULATION_SIZE; i++){

		Acum += (Pop[i].Fitness - Avg)*(Pop[i].Fitness - Avg);

	}
	double StdDev = sqrt(Acum/POPULATION_SIZE);

	errs() <<  generation << "	" << Min << "	" << Avg << "	" << Max << "	" << StdDev <<'\n';
}


void MemoryIdempotenceAnalysisImpl::computeCutSetWithGA(){

	//Build a map with the instructions in the paths
	std::map<Instruction*, bool> PathMap;
	for(unsigned int i = 0; i < AntidependencePaths_.size(); i++){
		for(unsigned int j = 0; j < AntidependencePaths_[i].size(); j++){
			PathMap[AntidependencePaths_[i][j]] = true;
		}
	}


	srand(RAND_SEED);

	//generate population
	generateRandomPopulation(F_, Population);
	fixInvalidIndividuals(AntidependencePaths_, PathMap, Population);

	for (int G = 0; G < NUM_GENERATIONS; G++){


		//errs() <<  "Generation " << G << '\n'<< '\n'<< '\n'<< '\n';

		//Fitness calculation
		updatePopulationFitness(Population);
		//printPopulation(Population);


		//Selection
		for(int i = 0; i < POPULATION_SIZE; i++){
			Individual Novo(tournament(Population));
			NextPopulation.push_back(Novo);
		}

		//Genetic operators
		applyGeneticOperators(NextPopulation);

		//Fix invalid individuals
		fixInvalidIndividuals(AntidependencePaths_, PathMap, NextPopulation);

		//New Population
		replacePopulation(Population, NextPopulation);

		if (PRINT_STATISTICS) printGenerationStatistics(G, Population);
	}

	//get the best element of the population

	//use the best element to insert the items into CutSet_
	// Example: CutSet_.insert(Instruction*);

	//Sort the members ascending by the fitness. The strongest element will be the last member.
	std::sort(Population.begin(), Population.end(), Individual::compare);
	Individual solution(Population[0]);

	//int score = 0;
        //errs()<<"solution:"<<"\n";
	for (unsigned int i = 0; i < solution.instructions.size(); i++){
		if (solution.instructions[i].isCut) {
			CutSet_.insert(solution.instructions[i].inst);
			//score++;
			//errs()<<(uint64_t)(solution.instructions[i].inst)<<" ";
			//errs()<<i<<"\n";
		}
	}
	//errs()<<"\n";
	//errs()<<"score:"<<score<<"\n";

	//printGenerationStatistics(NUM_GENERATIONS, Population);

}

////////////////////////////////////////////////////////////////////////////
//	END OF GA IMPLEMENTATION
////////////////////////////////////////////////////////////////////////////


static void dumpCandidate(const CandidateInfo &RedundantInfo,
                          const WorklistTy &Worklist) {
  dbgs() << "Redundant candidate in position ";
  WorklistTy::const_iterator It = std::lower_bound(
      Worklist.begin(),
      Worklist.end(),
      const_cast<CandidateInfo *>(&RedundantInfo),
      CandidateInfo::compare);
  while (*It != &RedundantInfo)
    ++It;
  dbgs() << (It - Worklist.begin() + 1) << "/" << Worklist.size();
  dbgs() << " " << getLocator(*RedundantInfo.getCandidate());
}

void MemoryIdempotenceAnalysisImpl::processRedundantCandidate(
    CandidateInfo *RedundantInfo,
    WorklistTy *Worklist,
    const AntidependencePathTy &Path) {
  DEBUG(dbgs() << "  Before: ";
        dumpCandidate(*RedundantInfo, *Worklist);
        dbgs() << "\n");

  // Find the place where the redundant candidate was in the worklist.  There
  // may be multiple candidates at the same priority so we may have to iterate
  // linearly a little bit.
  WorklistTy::iterator OldPosition = std::lower_bound(
    Worklist->begin(), Worklist->end(), RedundantInfo, CandidateInfo::compare);
  while (*OldPosition != RedundantInfo)
    ++OldPosition;

  // Remove the path and update the candidate's priority.  The worklist is now
  // no longer sorted.
  RedundantInfo->remove(Path);

  // Find the place to re-insert the redundant candidate in the worklist to make
  // it sorted again.
  WorklistTy::iterator NewPosition = std::lower_bound(
    Worklist->begin(), Worklist->end(), RedundantInfo, CandidateInfo::compare);
  assert(NewPosition <= OldPosition && "new position has higher priority");

  // Re-insert by rotation.
  std::rotate(NewPosition, OldPosition, next(OldPosition));
  DEBUG(dbgs() << "  After: ";
        dumpCandidate(*RedundantInfo, *Worklist);
        dbgs() << "\n");
}

void MemoryIdempotenceAnalysisImpl::print(raw_ostream &OS,
                                          const Module *M) const {
  OS << "\nMemoryIdempotenceAnalysis Cut Set:\n";
  for (MemoryIdempotenceAnalysis::const_iterator I = MIA_->begin(),
       E = MIA_->end(); I != E; ++I) {
    Instruction *Cut = *I;
    BasicBlock *CutBB = Cut->getParent();
    OS << "Cut at " << getLocator(*Cut) << " at loop depth "
      << LI_->getLoopDepth(CutBB) << "\n";
  }
  OS << "\n";
}

//===----------------------------------------------------------------------===//
// MemoryIdempotenceAnalysis
//===----------------------------------------------------------------------===//

char MemoryIdempotenceAnalysis::ID = 0;
INITIALIZE_PASS_BEGIN(MemoryIdempotenceAnalysis, "idempotence-analysis",
                "Idempotence Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(MemoryIdempotenceAnalysis, "idempotence-analysis",
                "Idempotence Analysis", true, true)

void MemoryIdempotenceAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AliasAnalysis>();
  AU.addRequired<DominatorTree>();
  AU.addRequired<LoopInfo>();
  AU.setPreservesAll();
}

bool MemoryIdempotenceAnalysis::doInitialization(Module &M) {
  Impl = new MemoryIdempotenceAnalysisImpl(this);
  CutSet_ = &Impl->CutSet_;
  return false;
}

bool MemoryIdempotenceAnalysis::doFinalization(Module &M) {
  delete Impl;
  return false;
}

void MemoryIdempotenceAnalysis::releaseMemory() {
  Impl->releaseMemory();
}

bool MemoryIdempotenceAnalysis::runOnFunction(Function &F) {
  return Impl->runOnFunction(F);
}

void MemoryIdempotenceAnalysis::print(raw_ostream &OS, const Module *M) const {
  Impl->print(OS, M);
}

