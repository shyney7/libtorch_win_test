#pragma once
#include <torch/torch.h>

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset>;