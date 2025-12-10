const Footer = () => (
  <footer className="bg-gray-900 text-white mt-8 py-4 flex flex-col items-center">
    <div className="text-sm mb-1">
      &copy; {new Date().getFullYear()} MineSafety – Designed for the modern mine worker.
    </div>
    <div className="text-xs">
      Built with React & Tailwind | For demo purposes only
    </div>
  </footer>
);

export default Footer;
