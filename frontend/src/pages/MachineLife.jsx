import React, { useState } from 'react';
import { motion } from 'framer-motion';

const equipmentList = ['Pump', 'Compressor', 'Turbine'];
const locationList = [
  'Delhi', 'Mumbai', 'Chennai', 'Hyderabad', 'Bangalore',
  'Kolkata', 'Jaipur', 'Ahmedabad', 'Pune', 'Goa'
];

function MachineLife() {
  const [inputs, setInputs] = useState({
    temperature: '',
    pressure: '',
    vibration: '',
    humidity: '',
    equipment: equipmentList[0],
    location: locationList[0]
  });
  const [result, setResult] = useState(null);

  const handleChange = (e) =>
    setInputs({ ...inputs, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    const res = await fetch('http://localhost:5000/predict_life', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputs)
    });
    setResult(await res.json());
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-100 via-white to-blue-200 px-6 py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="max-w-2xl w-full backdrop-blur-xl bg-white/70 rounded-3xl shadow-2xl border border-white/40 p-8"
      >
        <h2 className="text-3xl font-extrabold text-center text-transparent bg-clip-text bg-gray-900 mb-8">
          Equipment Fault Prediction (India)
        </h2>

        <form
          onSubmit={handleSubmit}
          className="space-y-5"
        >
          {['temperature', 'pressure', 'vibration', 'humidity'].map((field) => (
            <div key={field}>
              <label className="block font-semibold text-gray-800 capitalize mb-1">
                {field}
              </label>
              <input
                type="number"
                name={field}
                step="any"
                value={inputs[field]}
                onChange={handleChange}
                required
                placeholder={`Enter ${field}`}
                className="w-full rounded-lg border border-blue-300 focus:ring-2 focus:ring-blue-500 focus:outline-none px-4 py-2 bg-white/80 shadow-sm transition-all duration-300 hover:shadow-md"
              />
            </div>
          ))}

          <div>
            <label className="block font-semibold text-gray-800 mb-1">Equipment Type</label>
            <select
              name="equipment"
              value={inputs.equipment}
              onChange={handleChange}
              className="w-full rounded-lg border border-blue-300 focus:ring-2 focus:ring-blue-500 focus:outline-none px-4 py-2 bg-white/80 shadow-sm hover:shadow-md transition-all duration-300"
            >
              {equipmentList.map((e) => (
                <option key={e} value={e}>{e}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block font-semibold text-gray-800 mb-1">Location</label>
            <select
              name="location"
              value={inputs.location}
              onChange={handleChange}
              className="w-full rounded-lg border border-blue-300 focus:ring-2 focus:ring-blue-500 focus:outline-none px-4 py-2 bg-white/80 shadow-sm hover:shadow-md transition-all duration-300"
            >
              {locationList.map((l) => (
                <option key={l} value={l}>{l}</option>
              ))}
            </select>
          </div>

          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            type="submit"
            className="w-full py-3 font-semibold text-white bg-gradient-to-r from-blue-600 to-cyan-500 rounded-xl shadow-md hover:shadow-lg transition-all duration-300"
          >
            Predict
          </motion.button>
        </form>

        {result && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
            className={`mt-8 p-6 rounded-2xl border text-center ${
              result.fault_pred === 1
                ? 'bg-red-50 border-red-300 text-red-700'
                : 'bg-green-50 border-green-300 text-green-700'
            }`}
          >
            <h3 className="text-xl font-bold mb-2">
              {result.fault_pred === 1 ? '⚠️ Fault Detected' : '✅ Machine is Healthy'}
            </h3>
            <p className="text-lg">
              Probability:{" "}
              <b>{(result.probability * 100).toFixed(2)}%</b>
            </p>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}

export default MachineLife;
