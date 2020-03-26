#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <vector>
//#include "include/owndata.h"

//----------------------Funktionsprototyp Onelinevector----------------------

std::vector<float>
onelinevector(const std::vector<std::vector<float>> &invector);

//------------------------Funktionsprototyp csv2Dvector----------------------

std::vector<std::vector<float>> csv2Dvector(std::string inputFileName);

//-------------------Custom Dataset Class------------------------------------

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {

private:
std::vector<std::vector<float>> xdata, ydata;
torch::Tensor outputtensor, inputtensor;
public:
CustomDataset(std::string xpath, std::string ypath) {
    xdata = csv2Dvector(xpath);
    ydata = csv2Dvector(ypath);

    //create input tensor:
    unsigned int ivsize = xdata.size() * xdata.front().size();
    std::vector<float> olinevec(ivsize);
    olinevec = onelinevector(xdata);

    inputtensor = torch::from_blob(
      olinevec.data(), {static_cast<unsigned int>(xdata.size()),
                        static_cast<unsigned int>(xdata.front().size())});
    //create output tensor:
    unsigned int ovsize = ydata.size() * ydata.front().size();
    std::vector<float> iovec(ovsize);
    iovec = onelinevector(ydata);

    outputtensor = torch::from_blob(
      iovec.data(), {static_cast<unsigned int>(ydata.size()),
                        static_cast<unsigned int>(ydata.front().size())});
};

torch::data::Example<> get(size_t index) override {
    
    torch::Tensor sample_input = inputtensor[index];
    torch::Tensor sample_output = outputtensor[index];

    return {sample_input.clone(), sample_output.clone()};
};

// Return the length of data
torch::optional<size_t> size() const override {
    return ydata.size();
  };

};

size_t batch_size = 5;

//-----------------------------NETZDEFINITION-----------------------------
struct MeinNetz : torch::nn::Module {
  MeinNetz() {
    fc1 = register_module("fc1", torch::nn::Linear(10, 10));
    fc2 = register_module("fc2", torch::nn::Linear(10, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = fc2->forward(x);
    return x;
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};


//-------------------------------------main-Funktion-------------------------

int main() {
 
  auto custom_dataset = CustomDataset("input.csv","output.csv");
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(custom_dataset),batch_size);

  auto dataset_size = custom_dataset.size().value();
  int n_epochs = 50;

  auto net = std::make_shared<MeinNetz>();

  torch::optim::SGD optimizer(net->parameters(), 0.2);

  for(int epoch=1; epoch<=n_epochs; epoch++) {
    for(auto& batch: *data_loader) {
      auto data = batch.data;
      auto target = batch.target;

      data = data.to(torch::kF32);
      target = target.to(torch::kF32);

      optimizer.zero_grad();

      auto prediction = net->forward(data);
      auto loss = torch::mse_loss(prediction, target);

      loss.backward();

      optimizer.step();
      
      std::cout << "Epoch: " << epoch << " Loss: " 
      << loss.item<float>() << std::endl;
    }
  }

  return 0;
}

//--------------------------Funktionen--------------------------------------------

std::vector<float>
onelinevector(const std::vector<std::vector<float>> &invector) {

  std::vector<float> v1d;
  if (invector.size() == 0)
    return v1d;
  v1d.reserve(invector.size() * invector.front().size());

  for (auto &innervector : invector) {
    v1d.insert(v1d.end(), innervector.begin(), innervector.end());
  }

  return v1d;
}

//-------------------------csv2vector Funktionsdefinition--------------------------

std::vector<std::vector<float>> csv2Dvector(std::string inputFileName) {
  using namespace std;

  vector<vector<float>> data;
  ifstream inputFile(inputFileName);
  int l = 0;

  while (inputFile) {
    l++;
    string s;
    if (!getline(inputFile, s))
      break;
    if (s[0] != '#') {
      istringstream ss(s);
      vector<float> record;

      while (ss) {
        string line;
        if (!getline(ss, line, ','))
          break;
        try {
          record.push_back(stof(line));
        } catch (const std::invalid_argument e) {
          cout << "NaN found in file " << inputFileName << " line " << l
               << endl;
          e.what();
        }
      }

      data.push_back(record);
    }
  }

  if (!inputFile.eof()) {
    cerr << "Could not read file " << inputFileName << "\n";
    throw invalid_argument("File not found.");
  }

  return data;
}