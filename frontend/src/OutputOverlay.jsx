import React from 'react';

const OutputOverlay = ({ 
  open, 
  onOpenChange,
  results = [
    { name: "Detection Accuracy", description: "Wildfire detection confidence", value: "98%" },
    { name: "Affected Area", description: "Estimated fire spread area", value: "24 sq km" },
    { name: "Processing Time", description: "Analysis duration", value: "3.2 seconds" }
  ],
  processedImageUrl = "https://via.placeholder.com/400x320?text=Processed+Image"
}) => {
  if (!open) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold">Analysis Results</h2>
            <button 
              onClick={() => onOpenChange(false)}
              className="text-gray-500 hover:text-gray-700"
            >
              ✕
            </button>
          </div>

          <div className="space-y-6">
            {/* Results Visualization */}
            <div className="bg-gray-100 rounded-lg p-4">
              <h3 className="text-lg font-medium mb-3">Processed Image</h3>
              <div className="flex justify-center">
                <img 
                  src={processedImageUrl} 
                  alt="Processed image result" 
                  className="rounded-md max-w-full h-auto max-h-64 object-contain"
                />
              </div>
            </div>

            {/* Metrics */}
            <div className="bg-gray-100 rounded-lg p-4">
              <h3 className="text-lg font-medium mb-3">Detection Metrics</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {results.map((result, index) => (
                  <div key={index} className="bg-white p-4 rounded-md shadow">
                    <h4 className="font-bold text-lg">{result.name}</h4>
                    <p className="text-gray-600 text-sm mt-1">{result.description}</p>
                    <p className="text-blue-600 font-semibold mt-2">{result.value}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Analysis */}
            <div className="bg-gray-100 rounded-lg p-4">
              <h3 className="text-lg font-medium mb-3">Fire Analysis</h3>
              <div className="bg-white p-4 rounded-md shadow">
                <p className="mb-2">The analysis detected wildfire signatures with high confidence.</p>
                <p className="mb-2">Thermal patterns indicate active fire spread in the northwest quadrant.</p>
                <p className="mb-2">The system recommends immediate attention to the identified hotspots.</p>
              </div>
            </div>

            {/* Recommendations */}
            <div className="bg-gray-100 rounded-lg p-4">
              <h3 className="text-lg font-medium mb-3">Recommendations</h3>
              <div className="bg-white p-4 rounded-md shadow">
                <ul className="list-disc pl-5 space-y-2">
                  <li>Dispatch firefighting resources to the identified coordinates</li>
                  <li>Alert local authorities and evacuate nearby areas</li>
                  <li>Monitor wind direction for potential spread patterns</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OutputOverlay;