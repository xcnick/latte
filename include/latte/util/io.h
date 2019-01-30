#ifndef LATTE_UTIL_IO_H_
#define LATTE_UTIL_IO_H_

#include <google/protobuf/message.h>

#include "latte/proto/latte.pb.h"
#include "latte/common.h"

namespace latte {

using ::google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto);

// Read parameters from a file into a NetParameter proto message.
inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto)) << "Failed to parse file: " << filename;
}

inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

}

#endif  // LATTE_UTIL_IO_H_