// Mirrors the pinned gfx1201 Q6_K WMMA writeback invariant in mmq.cuh.
constexpr int kGfx1201WmmaWaves = 8;
constexpr int kQ6KWmmaOutputTileRows = 16;
constexpr int kBaselineMmqY = 128;
constexpr int kRequestedMmqY = 64;

static_assert(kGfx1201WmmaWaves * kQ6KWmmaOutputTileRows == kBaselineMmqY);
static_assert(kGfx1201WmmaWaves * kQ6KWmmaOutputTileRows != kRequestedMmqY);

int main() { return 0; }
