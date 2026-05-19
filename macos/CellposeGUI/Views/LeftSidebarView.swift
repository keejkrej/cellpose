import SwiftUI

struct LeftSidebarView: View {
    @Bindable var viewModel: MainViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                SidebarPanel(title: "Navigation") {
                    ForEach(SeriesState.axisOrder, id: \.self) { axis in
                        SeriesAxisRow(
                            label: SeriesState.axisLabels[axis, default: axis.uppercased()],
                            values: viewModel.seriesState.axisValues[axis] ?? [],
                            index: Binding(
                                get: { viewModel.seriesState.axisSliderIndices[axis, default: 0] },
                                set: { newValue in
                                    Task { await viewModel.setSeriesAxisIndex(axis, index: newValue) }
                                }
                            ),
                            isEnabled: viewModel.seriesState.isLoaded
                                && (viewModel.seriesState.axisValues[axis]?.count ?? 0) > 0,
                            onStep: { delta in
                                Task { await viewModel.navigateSeriesAxis(axis, delta: delta) }
                            }
                        )
                    }
                }
                .disabled(!viewModel.seriesState.isLoaded)

                SidebarPanel(title: "Views") {
                    Picker("View", selection: $viewModel.viewMode) {
                        ForEach(ViewMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                                .disabled(mode == .restored && !viewModel.hasRestoredView)
                        }
                    }
                    Toggle("norm3D", isOn: $viewModel.preprocessingParams.norm3D)
                    Button("auto saturation") {
                        Task { await viewModel.computeSaturation() }
                    }
                    .disabled(!viewModel.imageLoaded || viewModel.isBusy)
                    HStack {
                        Text("gray:")
                            .frame(width: 36, alignment: .leading)
                        Slider(value: $viewModel.displayParams.grayLow, in: 0 ... 255, step: 1)
                            .disabled(!viewModel.imageLoaded)
                        Slider(value: $viewModel.displayParams.grayHigh, in: 0 ... 255, step: 1)
                            .disabled(!viewModel.imageLoaded)
                    }
                }

                SidebarPanel(title: "Preprocessing") {
                    LabeledContent("sharpen radius") {
                        TextField("0", value: $viewModel.preprocessingParams.sharpenRadius, format: .number)
                            .frame(width: 60)
                    }
                    LabeledContent("smooth radius") {
                        TextField("0", value: $viewModel.preprocessingParams.smoothRadius, format: .number)
                            .frame(width: 60)
                    }
                    LabeledContent("tile norm blocksize") {
                        TextField("0", value: $viewModel.preprocessingParams.tileNormBlocksize, format: .number)
                            .frame(width: 60)
                    }
                    LabeledContent("tile norm smooth3D") {
                        TextField("0", value: $viewModel.preprocessingParams.tileNormSmooth3D, format: .number)
                            .frame(width: 60)
                    }
                    Toggle("save restored/filtered image", isOn: $viewModel.saveRestoredImage)
                    HStack {
                        Button("reset") {
                            viewModel.clearRestore()
                        }
                        .disabled(!viewModel.imageLoaded)
                        Button("apply") {
                            Task { await viewModel.applyPreprocessing() }
                        }
                        .disabled(!viewModel.imageLoaded || viewModel.isBusy)
                    }
                }
            }
            .padding(.vertical, 8)
        }
    }
}

private struct SeriesAxisRow: View {
    let label: String
    let values: [String]
    @Binding var index: Int
    let isEnabled: Bool
    let onStep: (Int) -> Void

    var body: some View {
        HStack(spacing: 6) {
            Text(label)
                .frame(width: 16, alignment: .leading)
            Button("<") { onStep(-1) }
                .disabled(!isEnabled || index <= 0)
            Slider(
                value: Binding(
                    get: { Double(index) },
                    set: { index = Int($0.rounded()) }
                ),
                in: 0 ... Double(max(values.count - 1, 0)),
                step: 1
            )
            .disabled(!isEnabled)
            Button(">") { onStep(1) }
                .disabled(!isEnabled || index >= max(values.count - 1, 0))
        }
    }
}
