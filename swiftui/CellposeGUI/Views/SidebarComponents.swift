import SwiftUI

struct SidebarPanel<Content: View>: View {
    let title: String
    @ViewBuilder var content: () -> Content

    var body: some View {
        GroupBox(title) {
            content()
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
    }
}
