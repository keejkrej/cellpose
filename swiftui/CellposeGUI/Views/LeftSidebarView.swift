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
                                && (viewModel.seriesState.axisValues[axis]?.count ?? 0) > 1,
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
                                .disabled(mode != .image)
                        }
                    }
                    Button("auto saturation") {
                        Task { await viewModel.computeSaturation() }
                    }
                    .disabled(!viewModel.imageLoaded || viewModel.isBusy)
                    SaturationSliderRow(
                        label: "min",
                        value: $viewModel.displayParams.grayLow,
                        isEnabled: viewModel.imageLoaded
                    )
                    SaturationSliderRow(
                        label: "max",
                        value: $viewModel.displayParams.grayHigh,
                        isEnabled: viewModel.imageLoaded
                    )
                    MaskBlendSliderRow(
                        value: $viewModel.displayParams.maskBlend,
                        isEnabled: viewModel.imageLoaded
                    )
                }

                SidebarPanel(title: "Drawing") {
                    Toggle("brush", isOn: $viewModel.brushMode)
                        .toggleStyle(.button)
                        .disabled(!viewModel.imageLoaded)
                        .onChange(of: viewModel.brushMode) { _, enabled in
                            if enabled {
                                viewModel.selectMode = false
                            } else {
                                viewModel.cancelStroke()
                            }
                        }
                    Toggle("select", isOn: $viewModel.selectMode)
                        .toggleStyle(.button)
                        .disabled(!viewModel.canUseLabelTools)
                        .onChange(of: viewModel.selectMode) { _, enabled in
                            if enabled {
                                viewModel.brushMode = false
                                viewModel.cancelStroke()
                            }
                        }
                    Button("delete") {
                        Task { await viewModel.deleteSelectedCells() }
                    }
                    .disabled(!viewModel.canUseLabelTools)
                    Button("edit") {
                        if !viewModel.prepareEditSelectedCells() {
                            viewModel.errorMessage = "Select one or more cells first."
                        }
                    }
                    .disabled(!viewModel.canUseLabelTools)
                    HStack {
                        Text("default class")
                        TextField("0", value: $viewModel.defaultClassID, format: .number)
                            .frame(width: 60)
                    }
                }
                .alert("Edit class", isPresented: $viewModel.showEditClassDialog) {
                    TextField("Class ID", value: $viewModel.editClassInitialValue, format: .number)
                    Button("OK") {
                        viewModel.applyClassToSelectedCells(classID: viewModel.editClassInitialValue)
                    }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("Class ID for selected cell(s)")
                }
            }
            .padding(.vertical, 8)
        }
    }
}

private struct SaturationSliderRow: View {
    let label: String
    @Binding var value: Double
    let isEnabled: Bool

    var body: some View {
        HStack(spacing: 6) {
            Text(label)
                .frame(width: 28, alignment: .leading)
            Slider(value: $value, in: 0 ... 255)
                .disabled(!isEnabled)
        }
    }
}

private struct MaskBlendSliderRow: View {
    @Binding var value: Double
    let isEnabled: Bool

    var body: some View {
        HStack(spacing: 6) {
            Text("blend")
                .frame(width: 28, alignment: .leading)
                .help("0 = image only, 1 = mask only")
            Text("img")
                .font(.caption2)
                .foregroundStyle(.secondary)
            Slider(value: $value, in: 0 ... 1, step: 0.01)
                .disabled(!isEnabled)
            Text("mask")
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(String(format: "%.2f", value))
                .frame(width: 36, alignment: .trailing)
                .monospacedDigit()
        }
    }
}

private struct SeriesAxisRow: View {
    let label: String
    let values: [String]
    @Binding var index: Int
    let isEnabled: Bool
    let onStep: (Int) -> Void

    private var maxIndex: Int {
        max(values.count - 1, 0)
    }

    private var sliderUpperBound: Double {
        Double(max(maxIndex, 1))
    }

    var body: some View {
        HStack(spacing: 6) {
            Text(label)
                .frame(width: 16, alignment: .leading)
            Button("<") { onStep(-1) }
                .disabled(!isEnabled || index <= 0)
            Slider(
                value: Binding(
                    get: { Double(min(index, maxIndex)) },
                    set: { index = min(maxIndex, max(0, Int($0.rounded()))) }
                ),
                in: 0 ... sliderUpperBound
            )
            .disabled(!isEnabled)
            Button(">") { onStep(1) }
                .disabled(!isEnabled || index >= maxIndex)
        }
    }
}
