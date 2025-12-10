import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Home from "./pages/Home";
import RockAnalyzer from "./pages/RockAnalyser";
import MachineLife from "./pages/MachineLife";
import Footer from "./components/Fotter";

function App() {
  return (
    <Router>
      <div className="flex flex-col min-h-screen">
        <Navbar />
        <main className="flex-1 bg-gray-50">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/rock-analyzer" element={<RockAnalyzer />} />
            <Route path="/machine-life" element={<MachineLife />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

const Navbar = () => (
  <nav className="bg-gray-900 text-white px-6 py-3 shadow flex justify-between items-center">
    <div className="font-extrabold text-2xl tracking-tight">
      MineSafety
    </div>
    <div className="space-x-6 font-medium">
      <Link to="/" className="hover:underline">Home</Link>
      <Link to="/rock-analyzer" className="hover:underline">Rock Analyzer</Link>
      <Link to="/machine-life" className="hover:underline">Machine Life Checking</Link>
    </div>
  </nav>
);

export default App;
