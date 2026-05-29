import AppKit
import Foundation
import Observation

@MainActor
@Observable
final class MainViewModel {
    var filename: String?
    var image: ImageData?
    var masks: MaskData?
    var ncells: Int = 0
    var selectedCell: Int32 = 0
    var showMasks = true
    var showOutlines = true
    var imageLoaded = false
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
    var displayParams = DisplayParameters()
    var trainingParams = TrainingParameters.createDefault(modelSaveFolder: "~/.cellpose/models/custom")
    var seriesState = SeriesState()
    var instanceClasses = InstanceClasses()
    var classFilterText = ""
    var defaultClassID: Int32 = 0
    var brushMode = false
    var selectMode = false
    var selectedCells: [Int32] = []
    var selectedLabelRows: Set<Int> = []
    var selectionRevision = 0
    var labelsRowsRevision = 0
    var ellipseDiameters: [MaskMetricsService.EllipseDiameters] = []
    var showEditClassDialog = false
    var editClassInitialValue: Int32 = 0
    var inStroke = false
    var strokeRevision = 0
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
    var canUseLabelTools: Bool { imageLoaded && ncells > 0 }
    var currentStrokePoints: [[Double]] { currentStroke }

    private let ml: MlInferenceEngine
    private let sessionStore = CellposeSessionStore()
    private let seriesDiscovery = SeriesDiscoveryService()
    private let session = SessionState()

    private var seriesDataset: SeriesDatasetPayload?
    private var currentStroke: [[Double]] = []
    private var pendingStrokes: [[[Double]]] = []

    init(ml: MlInferenceEngine) {
        self.ml = ml
    }

    var filteredCellCount: Int {
        guard let filter = InstanceClasses.parseFilter(classFilterText) else { return ncells }
        return instanceClasses.values.filter { $0 == filter }.count
    }

    func applyLoadedImage(path: String, image: ImageData) {
        session.imagePath = path
        session.image = image
        session.masks = nil
        session.flows = []
        session.recomputeMasks = false
        session.seriesDataset = seriesDataset

        filename = path
        self.image = image
        masks = nil
        ncells = 0
        ellipseDiameters = []
        recomputeMasks = false
        selectedCell = 0
        setCellSelection([])
        instanceClasses.replace(ncells: 0)
        progress = 1
        imageLoaded = true
        updateSaturationFromImage()
    }

    func applyLoadedSession(_ loaded: LoadedSession) {
        session.imagePath = loaded.imagePath
        session.image = loaded.image
        session.masks = loaded.masks
        session.flows = loaded.flows
        session.recomputeMasks = loaded.recomputeMasks
        session.model = loaded.model
        session.segmentation = loaded.segmentation
        segmentationParams = loaded.segmentation

        filename = loaded.imagePath
        image = loaded.image
        masks = loaded.masks
        ncells = loaded.masks.labels.isEmpty ? 0 : Int(loaded.masks.labels.max() ?? 0)
        recomputeMasks = loaded.recomputeMasks
        instanceClasses.replace(ncells: ncells)
        refreshEllipseDiameters()
        setCellSelection([])
        progress = 1
        imageLoaded = true
        updateSaturationFromImage()
    }

    func applyMaskUpdate(_ updated: MaskData, appendedClassID: Int32? = nil) {
        let previousNcells = ncells
        let newNcells = updated.labels.isEmpty ? 0 : Int(updated.labels.max() ?? 0)
        instanceClasses.replace(ncells: newNcells, loaded: instanceClasses.values)
        if let appendedClassID, newNcells > previousNcells {
            instanceClasses.setClass(row: newNcells - 1, classID: appendedClassID)
        }
        let colored = MaskEditService.withClassColors(masks: updated, classIDs: instanceClasses.values)
        session.masks = colored
        masks = colored
        ncells = newNcells
        refreshEllipseDiameters()
        labelsRowsRevision += 1
    }

    func refreshEllipseDiameters() {
        guard let masks, ncells > 0 else {
            ellipseDiameters = []
            return
        }
        ellipseDiameters = MaskMetricsService.ellipseDiameters(
            labels: masks.labels,
            width: masks.width,
            height: masks.height,
            ncells: ncells
        )
    }

    func formattedMajorDiameter(row: Int) -> String {
        guard row >= 0, row < ellipseDiameters.count else { return "" }
        let value = ellipseDiameters[row].major
        guard value.isFinite else { return "" }
        return String(format: "%.1f", value)
    }

    func formattedMinorDiameter(row: Int) -> String {
        guard row >= 0, row < ellipseDiameters.count else { return "" }
        let value = ellipseDiameters[row].minor
        guard value.isFinite else { return "" }
        return String(format: "%.1f", value)
    }

    func isCellSelected(_ label: Int32) -> Bool {
        label > 0 && selectedCells.contains(label)
    }

    func setCellSelection(_ cells: [Int32]) {
        let normalized = Array(Set(cells.filter { $0 > 0 })).sorted()
        guard normalized != selectedCells else {
            selectionRevision += 1
            return
        }
        selectedCells = normalized
        selectedCell = normalized.first ?? 0
        selectedLabelRows = Set(normalized.map { Int($0) - 1 }.filter { $0 >= 0 })
        selectionRevision += 1
    }

    func selectedCellIndices() -> [Int32] {
        if !selectedCells.isEmpty {
            return selectedCells
        }
        if selectedCell > 0 {
            return [selectedCell]
        }
        return []
    }

    func selectCellAt(x: Int, y: Int, additive: Bool) {
        guard let masks else { return }
        let label = masks.label(at: x, y: y)
        if label > 0 {
            if additive {
                var cells = selectedCells
                if !cells.contains(label) {
                    cells.append(label)
                }
                setCellSelection(cells)
            } else {
                setCellSelection([label])
            }
            return
        }
        if !additive {
            setCellSelection([])
        }
    }

    func selectCellsInRect(x0: Int, y0: Int, x1: Int, y1: Int, additive: Bool) {
        guard let masks else { return }
        let filter = InstanceClasses.parseFilter(classFilterText)
        let cells = MaskEditService.cellsFullyInRect(
            labels: masks.labels,
            width: masks.width,
            height: masks.height,
            x0: x0,
            y0: y0,
            x1: x1,
            y1: y1,
            filterClassID: filter,
            classIDs: instanceClasses.values
        )
        if additive {
            setCellSelection(selectedCells + cells)
        } else {
            setCellSelection(cells)
        }
    }

    func deleteSelectedCells() async {
        let cells = selectedCellIndices().map(Int.init)
        guard !cells.isEmpty else { return }
        await removeCells(indices: cells)
    }

    func prepareEditSelectedCells() -> Bool {
        let cells = selectedCellIndices()
        guard !cells.isEmpty else { return false }
        let classIDs = cells.compactMap { idx -> Int32? in
            let row = Int(idx) - 1
            guard row >= 0, row < instanceClasses.values.count else { return nil }
            return instanceClasses.values[row]
        }
        editClassInitialValue = classIDs.count == 1 || Set(classIDs).count == 1 ? (classIDs.first ?? 0) : 0
        showEditClassDialog = true
        return true
    }

    func applyClassToSelectedCells(classID: Int32) {
        for idx in selectedCellIndices() {
            let row = Int(idx) - 1
            if row >= 0 {
                setInstanceClass(row: row, classID: classID)
            }
        }
        labelsRowsRevision += 1
    }

    func setInstanceClass(row: Int, classID: Int32) {
        instanceClasses.setClass(row: row, classID: classID)
        guard let currentMasks = session.masks else { return }
        let colored = MaskEditService.withClassColors(masks: currentMasks, classIDs: instanceClasses.values)
        session.masks = colored
        masks = colored
        labelsRowsRevision += 1
    }

    func applyInferenceResult(_ result: InferResult) {
        session.flows = result.flows
        session.recomputeMasks = result.recomputeMasks
        session.model = selectedModel
        session.segmentation = cloneSegmentationParams()
        recomputeMasks = result.recomputeMasks
        if let resultMasks = result.masks {
            applyMaskUpdate(resultMasks)
        }
    }

    private func cloneSegmentationParams() -> SegmentationParameters {
        var copy = SegmentationParameters()
        copy.diameter = segmentationParams.diameter
        copy.flowThreshold = segmentationParams.flowThreshold
        copy.cellprobThreshold = segmentationParams.cellprobThreshold
        copy.percentileLow = segmentationParams.percentileLow
        copy.percentileHigh = segmentationParams.percentileHigh
        copy.niter = segmentationParams.niter
        copy.minSize = segmentationParams.minSize
        return copy
    }

    private func saveSessionIfNeeded() {
        guard session.imagePath != nil, session.masks != nil else { return }
        session.segmentation = cloneSegmentationParams()
        session.model = selectedModel
        let path = sessionStore.defaultPath(imagePath: session.imagePath!)
        try? sessionStore.save(path: path, session: session)
    }

    func refreshModels() async {
        do {
            let listed = try await ml.listModels()
            let custom = listed.custom.filter { $0.lowercased() != "cpsam" }
            models = ["CPSAM"] + custom
        } catch {
            models = ["CPSAM"]
        }
    }

    func loadImagePanel() async {
        guard let url = pickFile(allowedTypes: ["tif", "tiff", "png", "jpg", "jpeg", "gif"]) else { return }
        await loadImage(path: url.path)
    }

    func loadImage(path: String) async {
        await runTask(message: "Loading image…") {
            let (loadedImage, companion) = try await self.loadImageWithOptionalCompanion(path: path)
            if let companion {
                self.applyLoadedSession(companion)
                self.statusMessage = "Loaded \(URL(fileURLWithPath: path).lastPathComponent) with segmentation"
            } else {
                self.applyLoadedImage(path: path, image: loadedImage)
                self.statusMessage = "Loaded \(URL(fileURLWithPath: path).lastPathComponent)"
            }
        }
    }

    private func loadImageWithOptionalCompanion(path: String) async throws -> (ImageData, LoadedSession?) {
        let loadedImage = try await Task.detached {
            try ImageLoaderService().load(path: path)
        }.value
        let sessionPath = sessionStore.defaultPath(imagePath: path)
        guard FileManager.default.fileExists(atPath: sessionPath) else {
            return (loadedImage, nil)
        }
        let companion = try? await Task.detached {
            try CellposeSessionStore().loadCompanion(sessionPath: sessionPath, imagePath: path, image: loadedImage)
        }.value
        return (loadedImage, companion)
    }

    func loadSeg(path: String) async {
        await runTask(message: "Loading segmentation…") {
            let loaded = try await Task.detached {
                try CellposeSessionStore().load(path: path, imageLoader: ImageLoaderService())
            }.value
            self.applyLoadedSession(loaded)
            self.statusMessage = "Loaded \(URL(fileURLWithPath: path).lastPathComponent)"
        }
    }

    func loadSeriesFolder(subfolderTemplate: String, filenameTemplate: String) async {
        guard let folderURL = pendingFolderURL else { return }
        lastSeriesSubfolderTemplate = subfolderTemplate
        lastSeriesFilenameTemplate = filenameTemplate
        await runTask(message: "Discovering series…") {
            let discovery = try await Task.detached {
                try SeriesDiscoveryService().discover(
                    folder: folderURL.path,
                    subfolderTemplate: subfolderTemplate,
                    filenameTemplate: filenameTemplate
                )
            }.value
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
                let (loadedImage, companion) = try await self.loadImageWithOptionalCompanion(path: record.path)
                if let companion {
                    self.applyLoadedSession(companion)
                } else {
                    self.applyLoadedImage(path: record.path, image: loadedImage)
                }
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
            let suggestion = try seriesDiscovery.suggestTemplates(folder: folder.path)
            return (suggestion.subfolderTemplate, suggestion.filenameTemplate)
        } catch {
            return (lastSeriesSubfolderTemplate, lastSeriesFilenameTemplate)
        }
    }

    func navigateSeriesAxis(_ axis: String, delta: Int) async {
        guard seriesState.isLoaded, let values = seriesState.axisValues[axis], !values.isEmpty else { return }
        let current = seriesState.axisSliderIndices[axis, default: 0]
        let next = max(0, min(values.count - 1, current + delta))
        guard next != current else { return }
        seriesState.axisSliderIndices[axis] = next
        await loadCurrentSeriesRecord()
    }

    func setSeriesAxisIndex(_ axis: String, index: Int) async {
        guard seriesState.isLoaded,
              let values = seriesState.axisValues[axis],
              values.indices.contains(index)
        else { return }
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
            let (loadedImage, companion) = try await self.loadImageWithOptionalCompanion(path: record.path)
            if let companion {
                self.applyLoadedSession(companion)
            } else {
                self.applyLoadedImage(path: record.path, image: loadedImage)
            }
            self.statusMessage = record.label
        }
    }

    private func loadCurrentSeriesRecord() async {
        guard let recordIndex = resolveCurrentSeriesRecordIndex() else { return }
        guard recordIndex != seriesState.recordIndex else { return }
        seriesState.recordIndex = recordIndex
        guard let record = seriesState.currentRecord else { return }
        await runTask(message: "Loading frame…") {
            let (loadedImage, companion) = try await self.loadImageWithOptionalCompanion(path: record.path)
            if let companion {
                self.applyLoadedSession(companion)
            } else {
                self.applyLoadedImage(path: record.path, image: loadedImage)
            }
            self.statusMessage = record.label
        }
    }

    private func resolveCurrentSeriesRecordIndex() -> Int? {
        guard let dataset = seriesDataset, let lookup = dataset.lookup else { return nil }
        let key = "\(axisValue("position", in: dataset))_\(axisValue("time", in: dataset))_\(axisValue("channel", in: dataset))_\(axisValue("z", in: dataset))"
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

    func refreshLabelsFilter() {}

    func runSegmentation() async {
        guard session.imagePath != nil else { return }
        await runTask(message: "Running segmentation…", showProgress: true) {
            self.progress = 0.1
            let result = try await self.ml.infer(
                imagePath: self.session.imagePath!,
                modelName: self.selectedModel,
                customModel: self.isCustomModel,
                params: self.segmentationParams
            )
            self.progress = 1
            self.applyInferenceResult(result)
            self.statusMessage = "Found \(result.ncells) cells"
            self.saveSessionIfNeeded()
        }
    }

    func recomputeFromThresholds() async {
        guard recomputeMasks, !session.flows.isEmpty else { return }
        await runTask(message: "Recomputing masks…") {
            let result = try await self.ml.recompute(flows: self.session.flows, params: self.segmentationParams)
            if let updated = result.masks {
                self.applyMaskUpdate(updated)
            }
            self.statusMessage = "Recomputed \(result.ncells) cells"
            self.saveSessionIfNeeded()
        }
    }

    func computeSaturation() async {
        guard imageLoaded else { return }
        updateSaturationFromImage()
        statusMessage = "Saturation \(Int(displayParams.grayLow))-\(Int(displayParams.grayHigh))"
    }

    private func updateSaturationFromImage() {
        guard let image else { return }
        let pixels = image.pixels
        guard !pixels.isEmpty else { return }
        let samples = stride(from: 0, to: pixels.count, by: max(image.channels, 1)).map { Double(pixels[$0]) }
        let sorted = samples.sorted()
        let lowIndex = Int((segmentationParams.percentileLow / 100.0) * Double(sorted.count - 1))
        let highIndex = Int((segmentationParams.percentileHigh / 100.0) * Double(sorted.count - 1))
        displayParams.grayLow = sorted[max(0, lowIndex)]
        displayParams.grayHigh = sorted[min(sorted.count - 1, highIndex)]
    }

    func saveResults() async {
        guard session.imagePath != nil, session.masks != nil else { return }
        await runTask(message: "Saving…") {
            let savePath = self.sessionStore.defaultPath(imagePath: self.session.imagePath!)
            try self.sessionStore.save(path: savePath, session: self.session)
            self.statusMessage = "Saved \(URL(fileURLWithPath: savePath).lastPathComponent)"
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
            try await self.ml.removeModel(name: name)
            await self.refreshModels()
            self.selectedModelIndex = 0
            self.statusMessage = "Removed model \(name)"
        }
    }

    func removeCell(at x: Int, y: Int, modifierFlags: NSEvent.ModifierFlags) {
        guard let masks, !selectMode else { return }
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
        setCellSelection([label])
    }

    func removeCells(indices: [Int]) async {
        guard let currentMasks = session.masks else { return }
        await runTask(message: "Removing cells…") {
            let updated = MaskEditService.removeCells(masks: currentMasks, indices: indices)
            self.applyMaskUpdate(updated)
            self.setCellSelection([])
            self.saveSessionIfNeeded()
        }
    }

    func mergeCells(source: Int, target: Int) async {
        guard let currentMasks = session.masks else { return }
        await runTask(message: "Merging cells…") {
            let updated = MaskEditService.mergeCells(masks: currentMasks, source: source, target: target)
            self.applyMaskUpdate(updated)
            self.saveSessionIfNeeded()
        }
    }

    func beginStroke(at x: Int, y: Int, z: Int = 0) {
        currentStroke = [[Double(z), Double(y), Double(x), 0]]
        pendingStrokes = []
        inStroke = true
        notifyStrokeChanged()
    }

    func continueStroke(at x: Int, y: Int, z: Int = 0) {
        guard inStroke, !currentStroke.isEmpty else { return }
        let last = currentStroke[currentStroke.count - 1]
        if Int(last[1]) == y, Int(last[2]) == x {
            return
        }
        currentStroke.append([Double(z), Double(y), Double(x), 0])
        notifyStrokeChanged()
    }

    func cancelStroke() {
        currentStroke = []
        pendingStrokes = []
        guard inStroke else { return }
        inStroke = false
        notifyStrokeChanged()
    }

    func commitStroke() {
        guard !currentStroke.isEmpty else { return }
        pendingStrokes.append(currentStroke)
        currentStroke = []
    }

    func completeStroke(at x: Int, y: Int, z: Int = 0) async {
        guard inStroke else { return }
        continueStroke(at: x, y: y, z: z)
        commitStroke()
        inStroke = false
        notifyStrokeChanged()
        await finishDrawing()
    }

    private func notifyStrokeChanged() {
        strokeRevision += 1
    }

    func finishDrawing() async {
        guard !pendingStrokes.isEmpty, let image else { return }
        let strokes = pendingStrokes
        pendingStrokes = []
        await runTask(message: "Adding cell…") {
            guard let updated = MaskEditService.addMaskFromStrokes(
                masks: self.session.masks,
                imageWidth: image.width,
                imageHeight: image.height,
                strokes: strokes,
                classID: self.defaultClassID
            ) else {
                return
            }
            self.applyMaskUpdate(updated, appendedClassID: self.defaultClassID)
            self.saveSessionIfNeeded()
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
            let result = try await self.ml.train(params: self.trainingParams)
            self.progress = 1
            self.statusMessage = "Trained model \(result.modelName)"
            await self.refreshModels()
        }
    }

    func addCustomModel() async {
        guard let url = pickFile(allowedTypes: ["pth", "pt"]) else { return }
        await runTask(message: "Adding model…") {
            let name = try await self.ml.addModel(path: url.path)
            await self.refreshModels()
            if let index = self.models.firstIndex(of: name) {
                self.selectedModelIndex = index
            }
            self.statusMessage = "Added model \(name)"
        }
    }

    func handleDroppedURLs(_ urls: [URL]) async {
        guard let url = urls.first else { return }
        if url.pathExtension.lowercased() == "npy", url.lastPathComponent.hasSuffix("_seg.npy") {
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
        defer { isBusy = false }
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
