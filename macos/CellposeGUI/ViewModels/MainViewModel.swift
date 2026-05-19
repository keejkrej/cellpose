import AppKit
import Foundation
import Observation

@MainActor
@Observable
final class MainViewModel {
    var sessionID: String?
    var filename: String?
    var image: ImageData?
    var masks: MaskData?
    var ncells: Int = 0
    var selectedCell: Int32 = 0
    var showMasks = true
    var showOutlines = true
    var autosave = true
    var autoloadMasks = false
    var disableAutosave = false
    var saveRestoredImage = true
    var imageLoaded = false
    var hasRestoredView = false
    var showLoadFolderSheet = false
    var showTrainDialog = false
    var pendingFolderURL: URL?
    var lastSeriesSubfolderTemplate = ""
    var lastSeriesFilenameTemplate = "img_{t}_{c}_{z}.jpg"
    var isBusy = false
    var progress: Double = 0
    var statusMessage = "Ready"
    var errorMessage: String?
    var recomputeMasks = false

    var segmentationParams = SegmentationParameters()
    var preprocessingParams = PreprocessingParameters()
    var displayParams = DisplayParameters()
    var trainingParams = TrainingParameters.createDefault(modelSaveFolder: "~/.cellpose/models/custom")
    var seriesState = SeriesState()
    var instanceClasses = InstanceClasses()
    var classFilterText = ""
    var defaultClassID: Int32 = 0
    var viewMode: ViewMode = .image

    var models: [String] = ["CPSAM"]
    var selectedModelIndex = 0
    var selectedModel: String {
        let name = models.indices.contains(selectedModelIndex) ? models[selectedModelIndex] : "CPSAM"
        return name.lowercased() == "cpsam" ? "cpsam" : name
    }
    var isCustomModel: Bool { selectedModelIndex > 0 && models.first?.uppercased() == "CPSAM" }

    var canSaveMasks: Bool { imageLoaded && ncells > 0 }
    var canRunSegmentation: Bool { imageLoaded && !isBusy }

    private let engine: SegmentationEngine
    private var undoRemovedCells: [(Int32, [Int32])] = []
    private var currentStroke: [[Double]] = []
    private var pendingStrokes: [[[Double]]] = []
    private var seriesDataset: SeriesDatasetPayload?

    init(engine: SegmentationEngine) {
        self.engine = engine
    }

    var filteredCellCount: Int {
        guard let filter = InstanceClasses.parseFilter(classFilterText) else { return ncells }
        return instanceClasses.values.filter { $0 == filter }.count
    }

    func applyResult(_ result: SegmentationResult) {
        sessionID = result.sessionID
        filename = result.filename
        image = result.image
        masks = result.masks
        ncells = result.ncells
        recomputeMasks = result.recomputeMasks
        instanceClasses.replace(ncells: ncells)
        progress = 1
        imageLoaded = true
    }

    func refreshModels() async {
        do {
            let listed = try await engine.listModels()
            let custom = listed.custom.filter { $0.lowercased() != "cpsam" }
            models = ["CPSAM"] + custom
        } catch {
            models = ["CPSAM"]
        }
    }

    func loadImagePanel() async {
        guard let url = pickFile(allowedTypes: ["tif", "tiff", "png", "jpg", "jpeg", "gif", "npy"]) else { return }
        await loadImage(path: url.path)
    }

    func loadImage(path: String) async {
        await runTask(message: "Loading image…") {
            let result = try await self.engine.loadImage(path: path, load3D: false)
            self.applyResult(result)
            self.statusMessage = "Loaded \(URL(fileURLWithPath: path).lastPathComponent)"
        }
    }

    func loadSegPanel() async {
        guard let url = pickFile(allowedTypes: ["npy"]) else { return }
        await loadSeg(path: url.path)
    }

    func loadSeg(path: String) async {
        await runTask(message: "Loading segmentation…") {
            let result = try await self.engine.loadSeg(path: path, load3D: false)
            self.applyResult(result)
            self.statusMessage = "Loaded \(URL(fileURLWithPath: path).lastPathComponent)"
        }
    }

    func loadSeriesFolder(subfolderTemplate: String, filenameTemplate: String) async {
        guard let folderURL = pendingFolderURL else { return }
        lastSeriesSubfolderTemplate = subfolderTemplate
        lastSeriesFilenameTemplate = filenameTemplate
        await runTask(message: "Discovering series…") {
            let discovery = try await self.engine.discoverSeries(
                folder: folderURL.path,
                subfolderTemplate: subfolderTemplate,
                filenameTemplate: filenameTemplate
            )
            self.seriesDataset = discovery.dataset
            self.seriesState.folder = discovery.folder
            self.seriesState.subfolderTemplate = subfolderTemplate
            self.seriesState.filenameTemplate = filenameTemplate
            self.seriesState.records = discovery.records.map {
                SeriesRecord(index: $0.index, label: $0.label, path: $0.path)
            }
            self.seriesState.axisValues = discovery.axes
            self.seriesState.axisSliderIndices = SeriesState.axisOrder.reduce(into: [:]) { result, axis in
                result[axis] = 0
            }
            if let recordIndex = self.resolveCurrentSeriesRecordIndex() {
                self.seriesState.recordIndex = recordIndex
                let record = discovery.records[recordIndex]
                let result = try await self.engine.loadImage(path: record.path, load3D: false)
                self.applyResult(result)
            }
            self.statusMessage = "Loaded series with \(discovery.recordCount) records"
        }
    }

    func presentLoadFolderPanel() {
        guard let folder = pickFolder() else { return }
        pendingFolderURL = folder
        showLoadFolderSheet = true
    }

    func fetchSeriesTemplateSuggestions(for folder: URL) async -> (subfolder: String, filename: String) {
        do {
            let suggestion = try await engine.suggestSeriesTemplates(folder: folder.path)
            return (suggestion.subfolderTemplate, suggestion.filenameTemplate)
        } catch {
            return (lastSeriesSubfolderTemplate, lastSeriesFilenameTemplate)
        }
    }

    func navigateSeriesAxis(_ axis: String, delta: Int) async {
        guard seriesState.isLoaded, var values = seriesState.axisValues[axis], !values.isEmpty else { return }
        let current = seriesState.axisSliderIndices[axis, default: 0]
        let next = max(0, min(values.count - 1, current + delta))
        guard next != current else { return }
        seriesState.axisSliderIndices[axis] = next
        await loadCurrentSeriesRecord()
    }

    func setSeriesAxisIndex(_ axis: String, index: Int) async {
        guard seriesState.isLoaded, let values = seriesState.axisValues[axis], values.indices.contains(index) else {
            return
        }
        seriesState.axisSliderIndices[axis] = index
        await loadCurrentSeriesRecord()
    }

    func navigateSeries(delta: Int) async {
        guard seriesState.isLoaded else { return }
        let newIndex = max(0, min(seriesState.records.count - 1, seriesState.recordIndex + delta))
        guard newIndex != seriesState.recordIndex else { return }
        seriesState.recordIndex = newIndex
        syncSeriesSlidersToRecordIndex()
        guard let record = seriesState.currentRecord else { return }
        await runTask(message: "Loading frame…") {
            let result = try await self.engine.loadImage(path: record.path, load3D: false)
            self.applyResult(result)
            self.statusMessage = record.label
        }
    }

    private func loadCurrentSeriesRecord() async {
        guard let recordIndex = resolveCurrentSeriesRecordIndex() else { return }
        guard recordIndex != seriesState.recordIndex else { return }
        seriesState.recordIndex = recordIndex
        guard let record = seriesState.currentRecord else { return }
        await runTask(message: "Loading frame…") {
            let result = try await self.engine.loadImage(path: record.path, load3D: false)
            self.applyResult(result)
            self.statusMessage = record.label
        }
    }

    private func resolveCurrentSeriesRecordIndex() -> Int? {
        guard let dataset = seriesDataset, let lookup = dataset.lookup else { return nil }
        let position = axisValue("position", in: dataset)
        let time = axisValue("time", in: dataset)
        let channel = axisValue("channel", in: dataset)
        let z = axisValue("z", in: dataset)
        let key = "\(position)_\(time)_\(channel)_\(z)"
        return lookup[key]
    }

    private func axisValue(_ axis: String, in dataset: SeriesDatasetPayload) -> String {
        guard let values = dataset.axes[axis],
              let index = seriesState.axisSliderIndices[axis],
              values.indices.contains(index)
        else { return "" }
        return values[index]
    }

    private func syncSeriesSlidersToRecordIndex() {
        guard let dataset = seriesDataset,
              seriesState.records.indices.contains(seriesState.recordIndex)
        else { return }
        let record = dataset.records[seriesState.recordIndex]
        for axis in SeriesState.axisOrder {
            if let values = dataset.axes[axis],
               let index = dataset.axisIndex?[axis]?[recordValue(record, axis: axis)]
            {
                seriesState.axisSliderIndices[axis] = index
            } else if let values = dataset.axes[axis],
                      let valueIndex = values.firstIndex(of: recordValue(record, axis: axis))
            {
                seriesState.axisSliderIndices[axis] = valueIndex
            }
        }
    }

    private func recordValue(_ record: SeriesDatasetRecord, axis: String) -> String {
        switch axis {
        case "position": record.position
        case "time": record.time
        case "channel": record.channel
        default: record.z
        }
    }

    func refreshInstanceFilter() {
        // Triggers canvas redraw when class filter changes.
    }

    func runSegmentation() async {
        guard let sessionID else { return }
        await runTask(message: "Running segmentation…", showProgress: true) {
            self.progress = 0.1
            let result = try await self.engine.segment(
                sessionID: sessionID,
                imagePayload: nil,
                filename: self.filename,
                modelName: self.selectedModel,
                customModel: self.isCustomModel,
                params: self.segmentationParams,
                preprocess: self.preprocessingParams
            )
            self.progress = 1
            self.applyResult(result)
            self.statusMessage = "Found \(result.ncells) cells"
            if self.autosave && !self.disableAutosave {
                _ = try await self.engine.saveSeg(sessionID: result.sessionID, path: nil)
            }
        }
    }


    func recomputeFromThresholds() async {
        guard recomputeMasks, let sessionID else { return }
        await runTask(message: "Recomputing masks…") {
            let result = try await self.engine.recomputeMasks(sessionID: sessionID, params: self.segmentationParams)
            self.applyResult(result)
            self.statusMessage = "Recomputed \(result.ncells) cells"
        }
    }

    func applyPreprocessing() async {
        guard let sessionID else { return }
        await runTask(message: "Applying filter…") {
            let result = try await self.engine.preprocess(sessionID: sessionID, params: self.preprocessingParams)
            self.applyResult(result)
            self.hasRestoredView = true
            self.statusMessage = "Preprocessing applied"
        }
    }

    func clearRestore() {
        hasRestoredView = false
        if viewMode == .restored {
            viewMode = .image
        }
    }

    func computeSaturation() async {
        guard imageLoaded else { return }
        statusMessage = "Auto saturation not yet implemented"
    }

    func saveSeg() async {
        guard let sessionID else { return }
        await runTask(message: "Saving…") {
            let path = try await self.engine.saveSeg(sessionID: sessionID, path: nil)
            self.statusMessage = "Saved \(URL(fileURLWithPath: path).lastPathComponent)"
        }
    }

    func exportMasks() async {
        await exportWithPanel(defaultName: "_cp_masks.png", types: [.png, .tiff]) { [self] sessionID, path in
            let format = path.lowercased().contains("tif") ? "tif" : "png"
            return try await self.engine.exportMasks(sessionID: sessionID, path: path, format: format)
        }
    }

    func loadMasksPanel() async {
        guard let url = pickFile(allowedTypes: ["tif", "tiff", "png"]) else { return }
        statusMessage = "Load masks not yet wired for \(url.lastPathComponent)"
    }

    func exportOutlines() async {
        await exportWithPanel(defaultName: "_outline.txt", types: [.plainText]) { [self] sessionID, path in
            try await self.engine.exportOutlines(sessionID: sessionID, path: path)
        }
    }

    func exportFlows() async {
        await exportWithPanel(defaultName: "_flows.tif", types: [.tiff]) { [self] sessionID, path in
            try await self.engine.exportFlows(sessionID: sessionID, path: path)
        }
    }

    func exportROIs() async {
        await exportWithPanel(defaultName: "_rois.zip", types: [.zip]) { [self] sessionID, path in
            try await self.engine.exportROIs(sessionID: sessionID, path: path)
        }
    }

    func clearAllMasks() async {
        guard ncells > 0 else { return }
        await removeCells(indices: Array(1 ... ncells))
    }

    func undoAction() {}
    func undoRemoveAction() {}

    func removeSelectedModel() async {
        guard isCustomModel else { return }
        let name = selectedModel
        await runTask(message: "Removing model…") {
            try await self.engine.removeModel(name: name)
            await self.refreshModels()
            self.selectedModelIndex = 0
            self.statusMessage = "Removed model \(name)"
        }
    }

    private func exportWithPanel(
        defaultName: String,
        types: [UTType],
        export: @escaping (String, String) async throws -> String
    ) async {
        guard let sessionID, let filename else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = URL(fileURLWithPath: filename).deletingPathExtension().lastPathComponent + defaultName
        panel.allowedContentTypes = types
        guard panel.runModal() == .OK, let url = panel.url else { return }
        await runTask(message: "Exporting…") {
            let path = try await export(sessionID, url.path)
            self.statusMessage = "Exported \(URL(fileURLWithPath: path).lastPathComponent)"
        }
    }

    func removeCell(at x: Int, y: Int, modifierFlags: NSEvent.ModifierFlags) {
        guard let masks else { return }
        let label = masks.label(at: x, y: y)
        guard label > 0 else { return }

        if modifierFlags.contains(.option) {
            guard selectedCell > 0 else { return }
            Task { await mergeCells(source: Int(label), target: Int(selectedCell)) }
            return
        }

        if modifierFlags.contains(.control) {
            Task { await removeCells(indices: [Int(label)]) }
            return
        }

        selectedCell = label
    }

    func removeCells(indices: [Int]) async {
        guard let sessionID else { return }
        await runTask(message: "Removing cells…") {
            let result = try await self.engine.removeCells(sessionID: sessionID, indices: indices)
            self.applyResult(result)
            self.selectedCell = 0
            if self.autosave && !self.disableAutosave {
                _ = try await self.engine.saveSeg(sessionID: result.sessionID, path: nil)
            }
        }
    }

    func mergeCells(source: Int, target: Int) async {
        guard let sessionID else { return }
        await runTask(message: "Merging cells…") {
            let result = try await self.engine.mergeCells(sessionID: sessionID, source: source, target: target)
            self.applyResult(result)
        }
    }

    func beginStroke(at x: Int, y: Int, z: Int = 0) {
        currentStroke = [[Double(z), Double(y), Double(x), 0]]
    }

    func continueStroke(at x: Int, y: Int, z: Int = 0) {
        currentStroke.append([Double(z), Double(y), Double(x), 0])
    }

    func commitStroke() async {
        guard !currentStroke.isEmpty else { return }
        pendingStrokes.append(currentStroke)
        currentStroke = []
    }

    func finishDrawing() async {
        guard let sessionID, !pendingStrokes.isEmpty else { return }
        let strokes = pendingStrokes
        pendingStrokes = []
        await runTask(message: "Adding cell…") {
            let result = try await self.engine.addMask(
                sessionID: sessionID,
                strokes: strokes,
                classID: self.defaultClassID
            )
            self.applyResult(result)
            if self.autosave && !self.disableAutosave {
                _ = try await self.engine.saveSeg(sessionID: result.sessionID, path: nil)
            }
        }
    }

    func cycleViewMode(forward: Bool) {
        let modes = ViewMode.allCases
        guard let index = modes.firstIndex(of: viewMode) else { return }
        let next = forward ? (index + 1) % modes.count : (index + modes.count - 1) % modes.count
        viewMode = modes[next]
    }

    func trainModel() async {
        await runTask(message: "Training model…", showProgress: true) {
            self.progress = 0.2
            let result = try await self.engine.train(params: self.trainingParams)
            self.progress = 1
            self.statusMessage = "Trained model \(result.modelName)"
            await self.refreshModels()
        }
    }

    func addCustomModel() async {
        guard let url = pickFile(allowedTypes: ["pth", "pt", "torch"]) else { return }
        await runTask(message: "Adding model…") {
            let name = try await self.engine.addModel(path: url.path)
            await self.refreshModels()
            if let index = self.models.firstIndex(of: name) {
                self.selectedModelIndex = index
            }
            self.statusMessage = "Added model \(name)"
        }
    }

    func handleDroppedURLs(_ urls: [URL]) async {
        guard let url = urls.first else { return }
        if url.pathExtension.lowercased() == "npy", url.lastPathComponent.contains("_seg") {
            await loadSeg(path: url.path)
        } else {
            await loadImage(path: url.path)
        }
    }

    private func runTask(
        message: String,
        showProgress: Bool = false,
        operation: @escaping () async throws -> Void
    ) async {
        isBusy = true
        statusMessage = message
        errorMessage = nil
        if showProgress { progress = 0 }
        defer {
            isBusy = false
        }
        do {
            try await operation()
        } catch {
            errorMessage = error.localizedDescription
            statusMessage = "Error"
        }
    }

    private func pickFile(allowedTypes: [String]) -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = allowedTypes.compactMap { UTType(filenameExtension: $0) }
        return panel.runModal() == .OK ? panel.url : nil
    }

    private func pickFolder() -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        return panel.runModal() == .OK ? panel.url : nil
    }
}

import UniformTypeIdentifiers
