#ifdef USE_CUDNN
#include "latte/util/cudnn.h"

namespace latte {

namespace cudnn {

float dataType<float>::oneval = 1.0;
float dataType<float>::zeroval = 0.0;
const void *dataType<float>::one =
    reinterpret_cast<void *>(&dataType<float>::oneval);
const void *dataType<float>::zero =
    reinterpret_cast<void *>(&dataType<float>::zeroval);

double dataType<double>::oneval = 1.0;
double dataType<double>::zeroval = 0.0;
const void *dataType<double>::one =
    reinterpret_cast<void *>(&dataType<double>::oneval);
const void *dataType<double>::zero =
    reinterpret_cast<void *>(&dataType<double>::zeroval);

}  // namespace cudnn
}  // namespace latte

#endif