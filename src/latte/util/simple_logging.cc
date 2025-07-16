#include "latte/util/simple_logging.h"
#include <signal.h>

namespace latte {

void InitGoogleLogging(const char* argv0) {
  // Simple initialization - just store program name if needed
  // For now, we don't need to do anything special
}

void signal_handler(int sig) {
  std::cerr << "Signal " << sig << " received" << std::endl;
  std::abort();
}

void InstallFailureSignalHandler() {
  // Install signal handlers for common crash signals
  signal(SIGSEGV, signal_handler);
  signal(SIGABRT, signal_handler);
  signal(SIGFPE, signal_handler);
}

} // namespace latte
