#include <torch/torch.h>
#include <vector>
#include <string>

//----------------------Funktionsprototyp Onelinevector----------------------

std::vector<float>
onelinevector(const std::vector<std::vector<float>> &invector);

//------------------------Funktionsprototyp csv2Dvector----------------------

std::vector<std::vector<float>> csv2Dvector(std::string inputFileName);


class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {

private:
std::vector<std::vector<float>> xdata, ydata;
public:
CustomDataset(std::string xpath, std::string ypath) {
    xdata = csv2Dvector(xpath);
    ydata = csv2Dvector(ypath);
};

torch::data::Example<> get(size_t index) override {
    //create input tensor:
    unsigned int ovsize = xdata.size() * xdata.front().size();
    std::vector<float> olinevec(ovsize);
    olinevec = onelinevector(xdata);

    torch::Tensor inputtensor = torch::from_blob(
      olinevec.data(), {static_cast<unsigned int>(xdata.size()),
                        static_cast<unsigned int>(xdata.front().size())});
    //create output tensor:
    unsigned int ovsize = ydata.size() * ydata.front().size();
    std::vector<float> iovec(ovsize);
    iovec = onelinevector(ydata);

    torch::Tensor outputtensor = torch::from_blob(
      iovec.data(), {static_cast<unsigned int>(ydata.size()),
                        static_cast<unsigned int>(ydata.front().size())});


    return {inputtensor, outputtensor};
};

// Return the length of data
torch::optional<size_t> size() const override {
    return ydata.size();
  };

};

/* main
vector<string> list_images; // list of path of images
  vector<int> list_labels; // list of integer labels
 
  // Dataset init and apply transforms - None!
  auto custom_dataset = CustomDataset(list_images, list_labels).map(torch::data::transforms::Stack<>()); */

