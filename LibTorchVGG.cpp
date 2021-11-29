#include<torch/script.h>
#include<torch/torch.h>
#include<vector>
#include<string>
#include<io.h>
#include<opencv2/opencv.hpp>




// ProgramModule: dataset.h
// Traverse all .jpg pictures in the folder
void load_data_from_folder(std::string image_dir, std::string type, std::vector<std::string> &list_images, std::vector<int> &list_labels, int label);

class dataSetClc:public torch::data::Dataset<dataSetClc> {
public:
    int num_classes = 0;
    dataSetClc(std::string image_dir, std::string type) {
        load_data_from_folder(image_dir, std::string(type), image_paths, labels, num_classes-1);
    }
    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override{
        std::string image_path = image_paths.at(index);
        cv::Mat image = cv::imread(image_path);
        cv::resize(image, image, cv::Size(448, 448));
        int label = labels.at(index);
        torch::Tensor img_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte).permute({2, 0, 1}); // Channels x Height x Width
        torch::Tensor label_tensor = torch::full({1}, label);
        return {img_tensor.clone(), label_tensor.clone()};
    }
    // Override size() function, return the length of data
    torch::optional<size_t> size() const override {
        return image_paths.size();
    }
private:
    std::vector<std::string> image_paths;
    std::vector<int> labels;
};

// ProgramModule: vgg.h
inline torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size, int64_t stride = 1, int64_t padding = 0, bool with_bias = false) {
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
    conv_options.stride(stride);
    conv_options.padding(padding);
    conv_options.bias(with_bias);
    return conv_options;
}

// Hyperparameters for MaxPool2d
inline torch::nn::MaxPool2dOptions maxpool_options(int kernel_size, int stride) {
    torch::nn::MaxPool2dOptions maxpool_options(kernel_size);
    maxpool_options.stride(stride);
    return maxpool_options;
}

// Similar to the make_features function in pytorch, return a CNN instance which is a torch::nnL::Sequential objects
torch::nn::Sequential make_features(std::vector<int> &cfg, bool batch_norm);

// Dinifition for VGG, including initialization and forwarding
class VGGImpl: public torch::nn::Module {
private:
    torch::nn::Sequential features_{nullptr};
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Sequential classifier;
public:
    VGGImpl(std::vector<int> &cfg, int num_classes = 1000, bool batch_norm = false);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(VGG);

// ProgramModule: classification.h
class Classifier {
private:
    torch::Device device = torch::Device(torch::kCPU);
    VGG vgg = VGG{nullptr};
public:
    Classifier(int gpu_id = 0);
    void Initialize(int num_classes, std::string pretrained_path);
    void Train(int epochs, int batch_size, float learning_rate, std::string train_val_dir, std::string image_type, std::string save_path);
    int Predict(cv::Mat &image);
    void LoadWeight(std::string weight);
};





// ProgramModule: dataset.cpp
void load_data_from_folder(std::string path, std::string type, std::vector<std::string> &list_images, std::vector<int> &list_labels, int label) {
    long long hFile = 0;
    struct _finddata_t fileInfo;
    std::string pathName;
    if ((hFile = _findfirst(pathName.assign(path).append("/*.*").c_str(), &fileInfo)) == -1) {
        return;
        }
    do {
        const char* s = fileInfo.name;
        const char* t = type.data();
        
        // Sub-filefolder: True
        if (fileInfo.attrib&_A_SUBDIR) {
            // Traverse all files in this sub-filefolder 
            if (strcmp(s, ".") == 0 || strcmp(s, "..") == 0) continue;
            std::string sub_path = path + "/" + fileInfo.name;
            label++;
            load_data_from_folder(sub_path, type, list_images, list_labels, label);
            }
        // If it is .type file or not
        else {
            if (strstr(s, t)) {
                std::string image_path = path + "/" + fileInfo.name;
                list_images.push_back(image_path);
                list_labels.push_back(label);
            }
        }
    } while (_findnext(hFile, &fileInfo) == 0);
    return;
}


// ProgramModule: vgg.cpp
torch::nn::Sequential make_features(std::vector<int> &cfg, bool batch_norm) {
    torch::nn::Sequential features;
    int in_channels = 3;
    for(auto v : cfg) {
        if(v == -1) {
            features->push_back(torch::nn::MaxPool2d(maxpool_options(2, 2)));
        }
        else{
            auto conv2d = torch::nn::Conv2d(conv_options(in_channels, v, 3, 1, 1));
            features->push_back(conv2d);
            if(batch_norm) {
                features->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(v)));
            }
            features->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
            in_channels = v;
        }
    }
    return features;
}

VGGImpl::VGGImpl(std::vector<int> &cfg, int num_classes, bool batch_norm) {
    features_ = make_features(cfg, batch_norm);
    avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(7));
    classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(512 * 7 * 7, 4096)));
    classifier->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    classifier->push_back(torch::nn:Dropout());
    classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(4096, 4096)));
    classifier->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    classifier->push_back(torch::nn::Dropout());
    classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(4096, num_classes)));
    
    features_ = register_module("features", features_);
    classifier = register_module("classifier", classifier);
}

torch::Tensor VGGImpl::forward(torch::Tensor x) {
    x = features_->forward(x);
    x = avgpool(x);
    x = torch::flatten(x, 1);
    x = classifier->forward(x);
    return torch::log_softmax(x, 1);
}

// ProgramModule: Classification.cpp
Classifier::Classifier(int gpu_id) {
    if (gpu_id >= 0) {
        device = torch::Device(torch::kCUDA, gpu_id);
    }
    else {
        device = torch::Device(torch::kCPU);
    }
}

void Classifier::Initialize(int _num_classes, std::string pretrained_path) {
    std::vector<int> cfg_d = {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};
    auto net_pretrained = VGG(cfg_d, 1000, true);
    vgg = VGG(cfg_d, _num_classes, true);
    torch::load(net_pretrained, _pretrained_path);
    torch::OrderedDict<std::string, at::Tensor> pretrained_dict = net_pretrained->named_parameters();
    torch::OrderedDict<std::string, at::Tensor> model_dict = vgg->named_parameters();
    
    for (auto n = pretrained_dict.begin(); n != pretrained_dict.end(); n++) {
        if (strstr((*n).key().data(), "classifier")) {
            continue;
        }
        model_dict[(*n).key()] = (*n).value();
    }
    
    torch::autograd::GradMode::set_enabled(false); //enable the copy of the parameters
    auto new_params = model_dict;
    auto params = vgg->named_parameters(true /*recurse*/);
    auto buffers = vgg->named_buffers(true /*recurse*/);
    for (auto& val : new_params) {
        auto name = val.key();
        auto* t = params.find(name);
        if (t != nullptr) {
            t->copy_(val.value());
        }
        else {
            t = buffers.find(name);
            if (t != nullptr) {
                t->copy_(val.value());
            }
        }
    }
    torch::autograd::GradMode::set_enabled(true);
    try {
        vgg->to(device);
    }
    catch (const std::exception&e) {
        std::cout << e.what() << std::endl;
    }
    
    return;
}

void Classifier::Train(int epochs, int batch_size, float learning_rate, std::string train_val_dir, std::string image_type, std::string save_path) {
    std::string path_train = train_val_dir + "\\train";
    std::string path_val = train_val_dir + "\\val";
    
    auto custom_dataset_train = dataSetClc(path_train, image_type).map(torch::data::transforms::Stack<>());
    auto custom_dataset_val = dataSetClc(path_val, image_type).map(torch::data::transforms::Stack<>());
    
    auto data_loader_train = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_train), batch_size);
    auto data_loader_val = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_val), batch_size);
    
    float loss_train = 0;
    float loss_val = 0;
    float acc_train = 0.0;
    float acc_val = 0.0;
    float best_acc = 0.0;
    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
        size_t batch_index_train = 0;
        size_t batch_index_val = 0;
        if (epoch == int(num_epochs / 2)) { learning_rate /= 10; }
        torch::optim::Adam optimizer(vgg->parameters(), learning_rate); //Learning Rate
        if (epoch < int(num_epochs / 8)) {
            for (auto mm : vgg->named_parameters()) {
                if (strstr(mm.key().data(), "classifier")) {
                    mm.value().set_requires_grad(true);
                }
                else {
                    mm.value().set_requires_grad(false);
                }
            }
        }
        else {
            for (auto mm : vgg->named_parameters()) {
                mm.value().set_requires_grad(true);
            }
        }
        // Traverse data_loader,generate batches
        for (auto& batch : *data_loader_train) {
            auto data = batch.data;
            auto target = batch.target.squeeze();
            data = data.to(torch::kF32).to(device).div(255.0);
            target = target.to(torch::kInt64).to(device);
            optimizer.zero_grad();
            // Execute the model
            torch::Tensor prediction = vgg->forward(data);
            auto acc = prediction.argmax(1).eq(target).sum();
            acc_train += acc.template item<float>() / batch_size;
            // Compute the value of Loss
            torch::Tensor loss = torch::nll_loss(prediction, target);
            // Compute the gradient
            loss.backward();
            // Update the weights
            optimizer.step();
            loss_train += loss.item<float>();
            batch_index_train++;
            std::cout << "Epoch: " << epoch << " |Train Loss: " << loss_train / batch_index_train << " |Train Acc:" << acc_train / batch_index_train << "\r";
        }
        std::cout << std::endl;
        
        // Evaluation
        vgg->eval();
        for (auto& batch : *data_loader_val) {
            auto data = batch.data;
            auto target = batch.target.squeeze();
            data = data.to(torch::kF32).to(device).div(255.0);
            target = target.to(torch::kInt64).to(device);
            torch::Tensor prediction = vgg->forward(data);
            // Computing Loss, the cross entropy with NLL and Log_softmax
            torch::Tensor loss = torch::nll_loss(prediction, target);
            auto acc = prediction.argmax(1).eq(target).sum();
            acc_val += acc.template item<float>() / batch_size;
            loss_val += loss.item<float>();
            batch_index_val++;
            std::cout << "Epoch: " << epoch << " |Val Loss: " << loss_val / batch_index_val << " |Valid Acc " << acc_val / batch_index_val << "\r";
         }
        std::cout << std::endl;
        
        if (acc_val > best_acc) {
            torch::save(vgg, save_path);
            best_acc = acc_val;
        }
        loss_train = 0;
        loss_val = 0;
        acc_train = 0;
        acc_val = 0;
        batch_index_train = 0;
        batch_val = 0;
    }
}

int Classifier::Predict(cv::Mat& image) {
    cv::resize(image, image, cv::Size(448, 448));
    torch::Tensor img_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte).permute({2, 0, 1});
    img_tensor = img_tensor.to(device).unsqueeze(0).to(torch::kF32).div(255.0);
    auto prediction = vgg->forward(img_tensor);
    prediction = torch::softmax(prediction, 1);
    auto class_id = prediction.argmax(1);
    std::cout << pretiction << class_id;
    int ans = int(class_id.item().toInt());
    float prob = prediction[0][ans].item().toFloat();
    return ans;
}

void Classifier::LoadWeight(std::string weight) {
    torch::load(vgg, weight);
    vgg->eval();
    return;
}

// ProgramModule: main.cpp
int main(int argc, char *argv[]) {
    auto pavgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(7)); // output 7*7, the parameter can be a single integer n denoting n*n, or a tuple (h, w) denoting hieght * weight
    auto inp = torch::rand({1, 3, 7, 7});
    auto outp = pavgpool->forward(inp);
    std::cout << outp.sizes();
    
    std::vector<int> cfg_dd = {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};
    auto vgg_dd = VGG(cfg_dd, 1000, true);
    auto in = torch::rand({1, 3, 244, 244});
    auto dictdd = vgg_dd->named_parameters();
    vgg_dd->forward(in);
    for (auto n = dictdd.begin(); n != dictdd.end(); n++) {
        std::cout << (*n).key() << std::endl;
    }
    
    std::string vgg_path = "~/libt_proj/vgg3/vgg16_bn_pretrained.pt";
    std::string train_val_dir = "~/dataset/hymenoptera_data";
    Classifier classifier(0);
    classifier.Initialize(2, vgg_path);
    
    //predict
    classifier.LoadWeight("classifier_trained.pt");
    cv::Mat image = cv::imread(train_val_dir+"val/bees/2407809945_fb525ef54d.jpg");
    classifier.Predict(image);
    classifier.Train(300, 4, 0.0003, train_val_dir, ".jpg", "classifier_pretrained.pt");
    std::vector<int> cfg_a = {64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1};
    std::vector<int> cfg_d = {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};
    auto vgg = VGG(cfg_d, 1000, true);
    auto dict = vgg->named_parameters();
    torch::load(vgg, vgg_path);
    
    return;
}