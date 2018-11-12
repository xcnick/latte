#include <gflags/gflags.h>
#include <glog/logging.h>

#include "latte/latte.h"

DEFINE_string(
    gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "", "The solver definition protocol buffer text file.");
DEFINE_string(model, "", "The model definition protocol buffer text file.");
DEFINE_string(phase, "",
              "Optional; network phase (TRAIN or TEST). Only used for 'time'.");
DEFINE_int32(level, 0, "Optional; network level.");
DEFINE_string(stage, "",
              "Optional; network stages (not to be confused with phase), "
              "separated by ','.");
DEFINE_string(snapshot, "",
              "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
              "Optional; the pretrained weights to initialize finetuning, "
              "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50, "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
              "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
              "Optional; action to take when a SIGHUP signal is received: "
              "snapshot, stop or none.");

using BrewFunction = int (*)();
using BrewMap = std::map<std::string, BrewFunction>;
BrewMap g_brew_map;

#define RegisterBrewFunction(func)                       \
  namespace {                                            \
  class __Registerer_##func {                            \
   public:                                               \
    __Registerer_##func() { g_brew_map[#func] = &func; } \
  };                                                     \
  __Registerer_##func g_registerer_##func;               \
  }

static BrewFunction GetBrewFunction(const std::string &name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available latte actions:";
    for (BrewMap::iterator it = g_brew_map.begin(); it != g_brew_map.end();
         ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return nullptr;  // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
static void get_gpus(std::vector<int>* gpus) {
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS_gpu.size()) {
    std::vector<std::string> strings;
    latte::string_split(&strings, FLAGS_gpu, ",");
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(std::stoi(strings[i]));
    }
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  std::vector<int> gpus;
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    latte::Latte::SetDevice(gpus[i]);
    latte::Latte::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);


int main(int argc, char **argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(LATTE_VERSION));
  // Usage message.
  gflags::SetUsageMessage(
      "command line brew\n"
      "usage: latte <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  latte::GlobalInit(&argc, &argv);
  if (argc == 2) {
    return GetBrewFunction(std::string(argv[1]))();
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/latte");
  }
}
