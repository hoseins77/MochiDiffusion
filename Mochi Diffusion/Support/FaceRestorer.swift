//
//  FaceRestorer.swift
//  Mochi Diffusion
//
//  Created by Hossein Safaie on 7/17/23.
//

import CoreImage
import Foundation
import Vision

final class FaceRestorer {

    private let request: VNCoreMLRequest

    static var shared = FaceRestorer()

    init() {
        let config = MLModelConfiguration()
        /// Note: CPU & NE conflicts with Image Generation
        config.computeUnits = .cpuAndGPU

        /// Create a Vision instance using the image classifier's model instance
        guard let model = try? VNCoreMLModel(for: gfpgan(configuration: config).model) else {
            fatalError("Failed to create a `VNCoreMLModel` instance.")
        }

        /// Create an image classification request with an image classifier model
        request = VNCoreMLRequest(model: model) { request, _ in
            if let observations = request.results as? [VNClassificationObservation] {
                print(observations)
            }
        }

        self.request.imageCropAndScaleOption = .scaleFill /// output image's ratio will be fixed later
        self.request.usesCPUOnly = false
    }

    func restoreFace(cgImage: CGImage) async -> CGImage? {
        let handler = VNImageRequestHandler(cgImage: cgImage)
        let requests: [VNRequest] = [request]

        try? handler.perform(requests)
        guard let observation = self.request.results?.first as? VNPixelBufferObservation else { return nil }
        guard let pixelBuffer = resizePixelBuffer(
            observation.pixelBuffer,
            width: cgImage.width,
            height: cgImage.height
        ) else { return nil }
        return self.convertPixelBufferToCGImage(pixelBuffer: pixelBuffer)
    }

    func restoreFace(sdi: SDImage) async -> SDImage? {
        if !sdi.faceRestorationModel.isEmpty { return nil }
        guard let cgImage = sdi.image else { return nil }
        guard let restoredImage = await restoreFace(cgImage: cgImage) else { return nil }
        var restoredSdi = sdi
        restoredSdi.image = restoredImage
        restoredSdi.aspectRatio = CGFloat(Double(sdi.width) / Double(sdi.height))
        restoredSdi.faceRestorationModel = "GFPGAN"
        return restoredSdi
    }

    private func convertPixelBufferToCGImage(pixelBuffer: CVPixelBuffer) -> CGImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext(options: nil)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        return context.createCGImage(ciImage, from: CGRect(x: 0, y: 0, width: width, height: height))
    }
}
