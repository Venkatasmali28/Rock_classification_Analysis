import React, { useState } from "react";

const API_URL = "http://127.0.0.1:5000/predict";

function RockAnalyzer() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) return;

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("image", image);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setResult({ error: err.message || "Server error or cannot connect." });
    }
    setLoading(false);
  };

  return (
    <div className="max-w-xl mx-auto py-12 px-4">
      <h2 className="text-3xl font-bold mb-4 text-gray-900">Rock Analyzer</h2>
      <form
        onSubmit={handleSubmit}
        className="bg-white shadow p-6 rounded-md flex flex-col items-center gap-4"
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700"
        />
        {preview && (
          <img
            src={preview}
            alt="preview"
            className="w-56 h-56 object-contain rounded border"
          />
        )}
        <button
          type="submit"
          disabled={!image || loading}
          className="bg-blue-600 text-white rounded px-4 py-2 font-semibold hover:bg-blue-500 transition"
        >
          {loading ? "Analyzing..." : "Analyze Rock"}
        </button>
      </form>
      {result && (
        <div className="mt-8 bg-white p-6 rounded shadow text-gray-700">
          {result.error && (
            <div className="text-red-600 font-semibold">{result.error}</div>
          )}
          {result.result && (
            <>
              <div className="mb-2 text-xl font-semibold">
                Predicted class:{" "}
                <span className="text-blue-700">{result.result}</span>
              </div>
              <div className="mb-2">
                Confidence:{" "}
                <span className="font-bold">
                  {(result.confidence * 100).toFixed(2)}%
                </span>
              </div>
              {result.gradcam && (
                <div className="mb-4">
                  <div className="font-bold">Grad-CAM Heatmap:</div>
                  <img
                    src={`data:image/png;base64,${result.gradcam}`}
                    alt="Gradcam"
                    className="w-48 h-48 object-contain border rounded"
                  />
                </div>
              )}
              {result.segmentation_mask && (
                <div>
                  <div className="font-bold">Crack Segmentation Mask:</div>
                  <img
                    src={`data:image/png;base64,${result.segmentation_mask}`}
                    alt="Segmentation"
                    className="w-48 h-48 object-contain border rounded"
                  />
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default RockAnalyzer;
