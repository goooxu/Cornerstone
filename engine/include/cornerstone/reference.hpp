#pragma once

#include <cstdint>
#include <vector>

#include "cornerstone/board.hpp"

namespace cornerstone {

// 朴素参考着法生成，仅用于测试交叉比对。返回值按动作编号升序。
std::vector<int32_t> legal_moves_reference(const Board& b);

}  // namespace cornerstone
