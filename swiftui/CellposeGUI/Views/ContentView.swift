import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Bindable var viewModel: MainViewModel
    @State private var zoom: CGFloat = 1

    var body: some View {
        HSplitView {
            LeftSidebarView(viewModel: viewModel)
                .frame(minWidth: 220, idealWidth: 240, maxWidth: 320)

            VStack(spacing: 0) {
                ImageCanvasView(viewModel: viewModel, zoom: $zoom)
                    .background(Color(nsColor: .windowBackgroundColor))
                StatusBarView(viewModel: viewModel, zoom: zoom)
            }
            .frame(minWidth: 400)
            .layoutPriority(1)

            RightSidebarView(viewModel: viewModel)
                .frame(minWidth: 260, idealWidth: 280, maxWidth: 360)
        }
        .navigationTitle(
            viewModel.filename.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "Cellpose"
        )
        .alert(
            "Error",
            isPresented: Binding(
                get: { viewModel.errorMessage != nil },
                set: { if !$0 { viewModel.errorMessage = nil } }
            )
        ) {
            Button("OK") { viewModel.errorMessage = nil }
        } message: {
            Text(viewModel.errorMessage ?? "")
        }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            handleDrop(providers)
        }
        .sheet(isPresented: $viewModel.showLoadFolderSheet) {
            LoadFolderPatternSheet(viewModel: viewModel, isPresented: $viewModel.showLoadFolderSheet)
        }
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        for provider in providers {
            provider.loadItem(forTypeIdentifier: "public.file-url", options: nil) { item, _ in
                guard let data = item as? Data,
                      let url = URL(dataRepresentation: data, relativeTo: nil)
                else { return }
                Task { @MainActor in
                    await viewModel.handleDroppedURLs([url])
                }
            }
        }
        return true
    }
}

struct StatusBarView: View {
    @Bindable var viewModel: MainViewModel
    let zoom: CGFloat

    var body: some View {
        HStack {
            Text(viewModel.statusMessage)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            if viewModel.isBusy {
                ProgressView(value: viewModel.progress)
                    .frame(width: 120)
            }
            Text("\(viewModel.ncells) cells")
                .font(.caption.monospacedDigit())
            Text(String(format: "%.0f%%", zoom * 100))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.bar)
    }
}
