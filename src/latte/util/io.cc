#include "latte/util/io.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

namespace latte {

using google::protobuf::Message;
using google::protobuf::io::IstreamInputStream;

bool ReadProtoFromTextFile(const char *filename, Message *proto) {
  std::ifstream infile;
  infile.open(filename, std::ios::binary);
  CHECK_NE(infile.is_open(), false) << "File not found: " << filename;
  IstreamInputStream *input = new IstreamInputStream(&infile);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  infile.close();
  return success;
}

}  // namespace latte