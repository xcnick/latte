#include "latte/util/simple_logging.h"
#include <iostream>

#include "latte/latte.h"

// Simple command line argument storage
struct CommandLineArgs {
  std::string gpu = "";
  std::string model = "";
  int level = 0;
  std::string stage = "";
  std::string weights = "";
  int iterations = 50;
  std::string sigint_effect = "stop";
  std::string sighup_effect = "snapshot";
  bool alsologtostderr = true;
} FLAGS;

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
  if (FLAGS.gpu == "all") {
    int count = 0;
#ifdef USE_CUDA
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
  } else if (FLAGS.gpu.size()) {
    std::vector<std::string> strings;
    latte::string_split(&strings, FLAGS.gpu, ",");
    for (size_t i = 0; i < strings.size(); ++i) {
      gpus->push_back(std::stoi(strings[i]));
    }
  } else {
    CHECK_EQ(static_cast<int>(gpus->size()), 0);
  }
}

// latte commands to call by
//     latte <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS.gpu;
  std::vector<int> gpus;
  get_gpus(&gpus);
  for (size_t i = 0; i < gpus.size(); ++i) {
    latte::Latte::SetDevice(gpus[i]);
    latte::Latte::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);


void show_usage(const char* program_name) {
  std::cout << "command line brew\n"
            << "usage: " << program_name << " <command> <args>\n\n"
            << "commands:\n"
            << "  train           train or finetune a model\n"
            << "  test            score a model\n"
            << "  device_query    show GPU diagnostic information\n"
            << "  time            benchmark model execution time\n";
}

int main(int argc, char **argv) {
  // Run tool or show usage.
  latte::GlobalInit(&argc, &argv);
  if (argc == 2) {
    return GetBrewFunction(std::string(argv[1]))();
  } else {
    show_usage(argv[0]);
    return 1;
  }
}
