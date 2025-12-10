import React from "react";
import { motion } from "framer-motion";

function Home() {
  return (
    <div className="min-h-screen bg-gray-200 via-white to-blue-200 flex items-center justify-center px-6 py-12">
      <div className="max-w-4xl w-full backdrop-blur-xl bg-white/70 shadow-2xl rounded-3xl p-10 border border-white/40">
        <motion.h1
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-5xl font-extrabold mb-6 text-transparent bg-clip-text bg-gray-900 text-center"
        >
          Safety & Efficiency for Mine Workers
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="text-lg text-gray-700 text-center leading-relaxed mb-8"
        >
          <b className="text-black">MineSafety</b> is an intelligent assistant
          for enhancing safety, monitoring equipment health, and automating rock
          analysis for mining operations. Our goal is to make mining{" "}
          <span className="font-semibold text-gary-700">
            safer, more efficient, and more data-driven
          </span>{" "}
          for every worker and engineer.
        </motion.p>

        <div className="my-10 grid gap-8 md:grid-cols-2">
          <FeatureCard
            title="Rock Analyzer"
            description="Upload images to instantly classify rock types, detect cracks, and visualize weaknesses with AI-powered Grad-CAM heatmaps."
          />
          <FeatureCard
            title="Machine Life Checking"
            description="Monitor key machinery, predict failure, and optimize maintenance schedules. (Feature coming soon!)"
          />
        </div>

        <motion.div
          whileHover={{ scale: 1.02 }}
          className="bg-gradient-to-r from-blue-50 to-blue-100 p-6 rounded-2xl border border-blue-200 shadow-inner"
        >
          <span className="font-semibold text-gray-900 text-lg">
            How does MineSafety help?
          </span>
          <ul className="list-disc ml-6 mt-3 text-gray-700 space-y-1">
            <li>AI-based rock analysis enables safer, quicker geotechnical decisions</li>
            <li>Visualizes critical zones using Grad-CAM heatmaps</li>
            <li>Predicts machine degradation for proactive maintenance</li>
            <li>Reduces manual safety analysis errors with automation</li>
          </ul>
        </motion.div>
      </div>
    </div>
  );
}

function FeatureCard({ title, description }) {
  return (
    <motion.div
      whileHover={{
        scale: 1.05,
        boxShadow: "0 10px 30px rgba(37,99,235,0.25)",
      }}
      transition={{ type: "spring", stiffness: 300 }}
      className="bg-white/80 backdrop-blur-md shadow-md rounded-2xl p-6 border border-blue-200 hover:bg-blue-50 transition-all duration-300"
    >
      <h3 className="text-2xl font-bold mb-3 text-gray-900">{title}</h3>
      <p className="text-gray-700 leading-relaxed">{description}</p>
    </motion.div>
  );
}

export default Home;
