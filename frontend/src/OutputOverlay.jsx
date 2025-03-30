import React from 'react';

const OutputOverlay = ({ 
  open, 
  onOpenChange,
  message = "Images Processed Successfully",
  inputImages = [
    { name: "RGB Image", img_path: "data:image/jpeg;base64,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" },
    { name: "IR Image", img_path: "data:image/jpeg;base64,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" },
  ],
  results = [
    { name: "Fire Detected", description: "Is Fire Visible ? ", value: "Yes" },
    { name: "Smoke Detected", description: "Is Smoke Visible ? ", value: "No" }
  ],
}) => {
  if (!open) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold text-center">Wildfire Detection Results</h2>
            <button 
              onClick={() => onOpenChange(false)}
              className="text-gray-500 hover:text-gray-700"
            >
              ✕
            </button>
          </div>

          {results ? (
          <div>
            <div>
              <div className="bg-gray-100 rounded-lg p-4 mb-6">
                <h2 className="text-lg font-medium">{message}!</h2>
              </div>
            </div>

            <div className="space-y-6">
              <div className="bg-gray-100 rounded-lg p-4">
                <h3 className="text-lg font-medium mb-3 text-center">Input Images</h3>
                <div className="flex justify-around">
                  {inputImages.map((image, index) => (
                    
                    <div className="flex flex-col items-center">
                    <img 
                      key={index}
                      src={image.img_path} 
                      alt={image.name} 
                      className="rounded-md object-contain mx-4 shadow-neutral-600"
                    />
                    <p className="text-center">{image.name}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-gray-100 rounded-lg p-4">
                <h3 className="text-lg font-medium mb-3 text-center">Detection Metrics</h3>
                <div className="flex justify-center gap-10">
                  {results.map((result, index) => (
                    <div key={index} className="bg-white p-4 rounded-md shadow">
                      <h4 className="font-bold text-lg">{result.name}</h4>
                      <p className="text-gray-600 text-sm mt-1">{result.description}</p>
                      <p className="text-blue-600 font-semibold mt-2">{result.value}</p>
                    </div>
                  ))}
                  {/* {
                    results.map((result, index) => (
                      <div>{result}</div>
                    ))
                  } */}
                </div>
              </div>
            </div>
          </div>
          )
          :
          (<p>No results available</p>)}
        </div>
      </div>
    </div>
  );
};

export default OutputOverlay;