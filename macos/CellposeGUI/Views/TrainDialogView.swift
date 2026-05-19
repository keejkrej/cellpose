import SwiftUI

struct TrainDialogView: View {
    @Bindable var viewModel: MainViewModel
    @Binding var isPresented: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Train Settings")
                .font(.title2)

            TextField("Train data folder", text: $viewModel.trainingParams.trainDataFolder)
            TextField("Model name", text: $viewModel.trainingParams.modelName)
            TextField("Learning rate", value: $viewModel.trainingParams.learningRate, format: .number)
            TextField("Weight decay", value: $viewModel.trainingParams.weightDecay, format: .number)
            TextField("Epochs", value: $viewModel.trainingParams.nEpochs, format: .number)
            TextField("Save folder", text: $viewModel.trainingParams.modelSaveFolder)

            HStack {
                Spacer()
                Button("Cancel") { isPresented = false }
                Button("Train") {
                    isPresented = false
                    Task { await viewModel.trainModel() }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(viewModel.trainingParams.trainDataFolder.isEmpty)
            }
        }
        .padding(24)
        .frame(width: 480)
    }
}
