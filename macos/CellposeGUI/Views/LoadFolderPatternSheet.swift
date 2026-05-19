import SwiftUI

struct LoadFolderPatternSheet: View {
    @Bindable var viewModel: MainViewModel
    @Binding var isPresented: Bool

    @State private var subfolderTemplate = ""
    @State private var filenameTemplate = "img_{t}_{c}_{z}.jpg"

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Load Folder with Pattern")
                .font(.title2)

            Text("Use placeholders {t}, {p}, {c}, {z}. Subfolder matching is case-insensitive.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Form {
                TextField("Subfolder template", text: $subfolderTemplate)
                TextField("Filename template", text: $filenameTemplate)
            }
            .formStyle(.grouped)

            HStack {
                Spacer()
                Button("Cancel") { isPresented = false }
                Button("Load") {
                    isPresented = false
                    Task {
                        await viewModel.loadSeriesFolder(
                            subfolderTemplate: subfolderTemplate,
                            filenameTemplate: filenameTemplate
                        )
                    }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(filenameTemplate.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(24)
        .frame(width: 520)
        .task {
            if let folder = viewModel.pendingFolderURL {
                let suggested = await viewModel.fetchSeriesTemplateSuggestions(for: folder)
                subfolderTemplate = suggested.subfolder
                filenameTemplate = suggested.filename
            } else {
                subfolderTemplate = viewModel.lastSeriesSubfolderTemplate
                filenameTemplate = viewModel.lastSeriesFilenameTemplate
            }
        }
    }
}
