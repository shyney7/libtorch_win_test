#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <vector>
#include "include/owndata.h"

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

//----------------------Funktionsprototyp Onelinevector----------------------

std::vector<float>
onelinevector(const std::vector<std::vector<float>> &invector);

//------------------------Funktionsprototyp csv2Dvector----------------------

std::vector<std::vector<float>> csv2Dvector(std::string inputFileName);

//-------------------------------------main-Funktion-------------------------

int main() {
  /*
  std::vector<std::vector<float>> data = csv2Dvector("input.csv");

  /* for (auto l : data) {
    for (auto x : l)
    std::cout << x << " ";
    std::cout << std::endl;
  } 

  std::cout << "Importierte Daten aus der csv Datei (Rohformat): " << std::endl;
  for (int i = 0; i < data.size(); i++) {
    for (int j = 0; j < data.front().size(); j++) {
      std::cout << data[i][j] << " ";
    }
    std::cout << std::endl;
  }

  //-------------Transformation des Inputs in 1D für tensor integration:-------
  std::cout << "Transformiere Input Vector mit Zeilenanzahl: " << data.size()
            << " Und Spaltenanzahl: " << data.front().size()
            << " in einen Tensor:" << std::endl;

  unsigned int ivsize = data.size() * data.front().size();
  std::vector<float> linevec(ivsize);
  linevec = onelinevector(data);

  torch::Tensor itensor = torch::from_blob(
      linevec.data(), {static_cast<unsigned int>(data.size()),
                       static_cast<unsigned int>(data.front().size())});
  std::cout << "Input Tensor: \n" << itensor << std::endl;

  //------------------------Das gleiche für den Output (target)------------------
  std::vector<std::vector<float>> outputdata = csv2Dvector("output.csv");

  std::cout << "Transformiere Output Vector mit Zeilenanzahl: "
            << outputdata.size()
            << " Und Spaltenanzahl: " << outputdata.front().size()
            << " in einen Tensor:" << std::endl;

  unsigned int ovsize = outputdata.size() * outputdata.front().size();
  std::vector<float> olinevec(ovsize);
  olinevec = onelinevector(outputdata);

  torch::Tensor otensor = torch::from_blob(
      olinevec.data(), {static_cast<unsigned int>(outputdata.size()),
                        static_cast<unsigned int>(outputdata.front().size())});
  std::cout << "Output Tensor: \n" << otensor << std::endl;         */

  //-------------------------------------------------------------------------------
  auto custom_dataset = CustomDataset("input.csv","output.csv").map(torch::data::transforms::Stack<>());
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(custom_dataset),batch_size);

  // Objekt der Klasse MeinNetz erzeugen:
 // auto net = std::make_shared<MeinNetz>();

  // Optimierer Auswahl:
  //torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.2);
/*
  // Lernschleife:
  for (size_t epoch = 1; epoch <= 50; epoch++) {
    auto input = torch::autograd::Variable(itensor);
    auto output = torch::autograd::Variable(otensor);

    optimizer.zero_grad();
    torch::Tensor prediction = net->forward(input);
    torch::Tensor loss = torch::mse_loss(prediction, output);
    loss.backward();
    optimizer.step();

    std::cout << "Epoch: " << epoch << " Loss: " << loss.item<float>()
              << std::endl;
  }
*/
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