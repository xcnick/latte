#ifndef LATTE_UTIL_SIMPLE_LOGGING_H_
#define LATTE_UTIL_SIMPLE_LOGGING_H_

#include <iostream>
#include <sstream>
#include <cstdlib>

namespace latte {

// Simple logging levels
enum LogLevel {
  INFO = 0,
  WARNING = 1,
  ERROR = 2,
  FATAL = 3
};

// Simple logger class
class SimpleLogger {
 public:
  SimpleLogger(LogLevel level, const char* file, int line)
      : level_(level), file_(file), line_(line) {}

  ~SimpleLogger() {
    std::cerr << "[" << LevelToString(level_) << "] "
              << file_ << ":" << line_ << " " << stream_.str() << std::endl;
    if (level_ == FATAL) {
      std::abort();
    }
  }

  std::ostringstream& stream() { return stream_; }

 private:
  LogLevel level_;
  const char* file_;
  int line_;
  std::ostringstream stream_;

  const char* LevelToString(LogLevel level) {
    switch (level) {
      case INFO: return "INFO";
      case WARNING: return "WARNING";
      case ERROR: return "ERROR";
      case FATAL: return "FATAL";
      default: return "UNKNOWN";
    }
  }
};

// Check class for assertions
class CheckLogger {
 public:
  CheckLogger(const char* condition, const char* file, int line)
      : condition_(condition), file_(file), line_(line) {}

  ~CheckLogger() {
    std::cerr << "[FATAL] " << file_ << ":" << line_
              << " Check failed: " << condition_ << " " << stream_.str() << std::endl;
    std::abort();
  }

  std::ostringstream& stream() { return stream_; }

 private:
  const char* condition_;
  const char* file_;
  int line_;
  std::ostringstream stream_;
};

// Global initialization functions
void InitGoogleLogging(const char* argv0);
void InstallFailureSignalHandler();

} // namespace latte

// Logging macros
#define LOG(level) \
  latte::SimpleLogger(latte::level, __FILE__, __LINE__).stream()

// Debug logging - only enabled in debug builds
#ifdef NDEBUG
#define DLOG(level) \
  if (true) {} else std::ostringstream()
#else
#define DLOG(level) LOG(level)
#endif

#define CHECK(condition) \
  !(condition) ? latte::CheckLogger(#condition, __FILE__, __LINE__).stream() : std::ostringstream().flush()

#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_NE(a, b) CHECK((a) != (b))
#define CHECK_LT(a, b) CHECK((a) < (b))
#define CHECK_LE(a, b) CHECK((a) <= (b))
#define CHECK_GT(a, b) CHECK((a) > (b))
#define CHECK_GE(a, b) CHECK((a) >= (b))

// Namespace alias for compatibility - only for specific functions
namespace google {
  using latte::InitGoogleLogging;
  using latte::InstallFailureSignalHandler;
}

#endif // LATTE_UTIL_SIMPLE_LOGGING_H_
