import Foundation

enum MaskMetricsService {
    struct EllipseDiameters {
        let major: Double
        let minor: Double

        static let invalid = EllipseDiameters(major: .nan, minor: .nan)
    }

    static func ellipseDiameters(labels: [Int32], width: Int, height: Int, ncells: Int) -> [EllipseDiameters] {
        guard ncells > 0 else { return [] }

        var coordsByLabel = [Int32: [(y: Double, x: Double)]](minimumCapacity: ncells)
        for y in 0 ..< height {
            let rowOffset = y * width
            for x in 0 ..< width {
                let label = labels[rowOffset + x]
                guard label > 0 else { continue }
                coordsByLabel[label, default: []].append((Double(y), Double(x)))
            }
        }

        return (1 ... ncells).map { label in
            guard let coords = coordsByLabel[Int32(label)], coords.count >= 5 else {
                return .invalid
            }
            return equivalentEllipseDiameters(coords: coords)
        }
    }

    static func equivalentEllipseDiameters(coords: [(y: Double, x: Double)]) -> EllipseDiameters {
        let count = Double(coords.count)
        var yMean = 0.0
        var xMean = 0.0
        for coord in coords {
            yMean += coord.y
            xMean += coord.x
        }
        yMean /= count
        xMean /= count

        var mu20 = 0.0
        var mu02 = 0.0
        var mu11 = 0.0
        for coord in coords {
            let y = coord.y - yMean
            let x = coord.x - xMean
            mu20 += x * x
            mu02 += y * y
            mu11 += x * y
        }
        mu20 /= count
        mu02 /= count
        mu11 /= count

        let common = sqrt((mu20 - mu02) * (mu20 - mu02) + 4 * mu11 * mu11)
        let lambda1 = max((mu20 + mu02 + common) / 2, 0)
        let lambda2 = max((mu20 + mu02 - common) / 2, 0)
        let major = 4 * sqrt(lambda1)
        let minor = 4 * sqrt(lambda2)
        return EllipseDiameters(major: major, minor: minor)
    }
}
