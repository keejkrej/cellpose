import SwiftUI

struct RightSidebarView: View {
    @Bindable var viewModel: MainViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                SidebarPanel(title: "Segmentation") {
                    HStack {
                        Text("model:")
                        Picker("model", selection: $viewModel.selectedModelIndex) {
                            ForEach(Array(viewModel.models.enumerated()), id: \.offset) { index, model in
                                Text(model).tag(index)
                            }
                        }
                        .labelsHidden()
                    }
                    LabeledContent("diameter") {
                        TextField("0", value: $viewModel.segmentationParams.diameter, format: .number)
                            .frame(width: 70)
                    }
                    LabeledContent("flow threshold") {
                        TextField("0.4", value: $viewModel.segmentationParams.flowThreshold, format: .number)
                            .frame(width: 70)
                            .onChange(of: viewModel.segmentationParams.flowThreshold) {
                                Task { await viewModel.recomputeFromThresholds() }
                            }
                    }
                    LabeledContent("cellprob threshold") {
                        TextField("0", value: $viewModel.segmentationParams.cellprobThreshold, format: .number)
                            .frame(width: 70)
                            .onChange(of: viewModel.segmentationParams.cellprobThreshold) {
                                Task { await viewModel.recomputeFromThresholds() }
                            }
                    }
                    LabeledContent("norm percentile lower") {
                        TextField("1", value: $viewModel.segmentationParams.percentileLow, format: .number)
                            .frame(width: 70)
                    }
                    LabeledContent("norm percentile upper") {
                        TextField("99", value: $viewModel.segmentationParams.percentileHigh, format: .number)
                            .frame(width: 70)
                    }
                    LabeledContent("niter dynamics") {
                        TextField("0", value: $viewModel.segmentationParams.niter, format: .number)
                            .frame(width: 70)
                            .onChange(of: viewModel.segmentationParams.niter) {
                                Task { await viewModel.recomputeFromThresholds() }
                            }
                    }
                    HStack {
                        Button("run") {
                            Task { await viewModel.runSegmentation() }
                        }
                        .disabled(!viewModel.canRunSegmentation)
                        ProgressView(value: viewModel.progress)
                            .opacity(viewModel.isBusy ? 1 : 0)
                    }
                }

                SidebarPanel(title: "Labels table") {
                    HStack {
                        Text("class filter:")
                        TextField("all", text: $viewModel.classFilterText)
                            .onSubmit { viewModel.refreshLabelsFilter() }
                            .onChange(of: viewModel.classFilterText) { _, _ in
                                viewModel.refreshLabelsFilter()
                            }
                    }
                    LabelsTableView(viewModel: viewModel)
                        .frame(minHeight: 120)
                }
            }
            .padding(.vertical, 8)
        }
        .sheet(isPresented: $viewModel.showTrainDialog) {
            TrainDialogView(viewModel: viewModel, isPresented: $viewModel.showTrainDialog)
        }
    }
}

struct LabelsTableView: View {
    @Bindable var viewModel: MainViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("ROI").frame(width: 60, alignment: .leading).font(.caption.bold())
                Text("Class ID").frame(width: 60, alignment: .leading).font(.caption.bold())
                Text("Major diam.").frame(width: 72, alignment: .leading).font(.caption.bold())
                Text("Minor diam.").font(.caption.bold())
            }
            if viewModel.ncells == 0 {
                Text("")
                    .frame(height: 1)
            } else {
                List(viewModel.ncells > 0 ? Array(0 ..< viewModel.ncells) : [], id: \.self, selection: $viewModel.selectedLabelRows) { row in
                    HStack {
                        Text("\(row + 1)")
                            .frame(width: 60, alignment: .leading)
                        TextField(
                            "0",
                            value: Binding(
                                get: {
                                    row < viewModel.instanceClasses.values.count
                                        ? viewModel.instanceClasses.values[row]
                                        : 0
                                },
                                set: { newValue in
                                    viewModel.setInstanceClass(row: row, classID: newValue)
                                }
                            ),
                            format: .number
                        )
                        .frame(width: 60)
                        Text(viewModel.formattedMajorDiameter(row: row))
                            .frame(width: 72, alignment: .leading)
                            .foregroundStyle(.secondary)
                        Text(viewModel.formattedMinorDiameter(row: row))
                            .foregroundStyle(.secondary)
                    }
                }
                .frame(minHeight: 120)
                .onChange(of: viewModel.selectedLabelRows) { _, rows in
                    viewModel.setCellSelection(rows.sorted().map { Int32($0 + 1) })
                }
            }
        }
    }
}
