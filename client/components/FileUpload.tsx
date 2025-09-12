import React, { useState } from 'react';

const FileUploadPopup = ({ onSubmit, onClose }) => {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleSubmit = () => {
    onSubmit(selectedFile);
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-75 flex items-center justify-center z-50">
      <div className="file-upload-popup bg-white p-6 rounded shadow-lg">
        <div className="flex justify-between pb-6">
          <h1 className="text-xl md:text-2xl font-medium">File Upload</h1>
          <button onClick={onClose} className="text-2xl leading-8">&times;</button>
        </div>
        <input type="file" accept=".pdf" onChange={handleFileChange} />
        <div className="flex justify-end mt-4">
          <button className="mr-2 px-4 py-2 bg-gray-300 rounded" onClick={onClose}>Cancel</button>
          <button className="px-4 py-2 bg-blue-500 text-white rounded" onClick={handleSubmit}>Upload</button>
        </div>
      </div>
    </div>
  );
};

export default FileUploadPopup;
