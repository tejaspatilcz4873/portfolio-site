import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";

const sectionVariants = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6 } },
};

const PortfolioApp = () => {
  const [repos, setRepos] = useState([]);

  // 👉 Add repos you want to showcase here
  const repoList = [
    { user: "tejaspatil4873", name: "python-notes" },
    { user: "tejaspatil4873", name: "networking-handbook" },
    { user: "tejaspatil4873", name: "aws-data-pipelines" },
  ];

  useEffect(() => {
    Promise.all(
      repoList.map((repo) =>
        fetch(`https://api.github.com/repos/${repo.user}/${repo.name}`)
          .then((res) => res.json())
      )
    ).then((data) => setRepos(data));
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800">
      {/* Navbar */}
      <nav className="fixed top-0 left-0 w-full bg-white shadow-md z-50 flex justify-between items-center px-8 py-4">
        <h1 className="text-xl font-bold">Tejas Patil</h1>
        <div className="space-x-6">
          <a href="#about" className="hover:text-blue-500">About</a>
          <a href="#experience" className="hover:text-blue-500">Experience</a>
          <a href="#skills" className="hover:text-blue-500">Skills</a>
          <a href="#projects" className="hover:text-blue-500">Projects</a>
          <a href="#contact" className="hover:text-blue-500">Contact</a>
        </div>
      </nav>

      {/* Hero */}
      <motion.section
        className="h-screen flex flex-col justify-center items-center text-center"
        variants={sectionVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <h1 className="text-5xl font-bold">Tejas Patil</h1>
        <p className="mt-4 text-xl">
          AWS Data Engineer | MBA (Business Analytics) | Cloud & Data Solutions Builder
        </p>
        <div className="mt-6 space-x-4">
          <a href="/resume.pdf" className="px-4 py-2 bg-blue-600 text-white rounded-lg shadow hover:bg-blue-700">
            Download Résumé
          </a>
          <a href="#contact" className="px-4 py-2 bg-gray-200 rounded-lg shadow hover:bg-gray-300">
            Contact Me
          </a>
        </div>
      </motion.section>

      {/* About */}
      <motion.section
        id="about"
        className="p-20 max-w-3xl mx-auto text-center"
        variants={sectionVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <h2 className="text-3xl font-semibold">About Me</h2>
        <p className="mt-4 text-lg">
          I’m an AWS Data Engineer with 2.5+ years of experience building scalable pipelines
          and cloud-native data solutions. Currently pursuing an MBA in Business Analytics
          at SCMHRD, Pune (2024–2026). Skilled in AWS (S3, Glue, Redshift, RDS, Lambda, SageMaker),
          Snowflake, Python, and SQL, I enjoy solving data challenges and delivering
          business insights.
        </p>
      </motion.section>

      {/* Experience */}
      <motion.section
        id="experience"
        className="p-20 bg-gray-100"
        variants={sectionVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <h2 className="text-3xl font-semibold text-center mb-10">Experience</h2>
        <div className="space-y-8 max-w-4xl mx-auto">
          <div>
            <h3 className="text-xl font-bold">AWS Data Engineer – Capgemini</h3>
            <p className="text-gray-700">Aug 2021 – Oct 2023</p>
            <ul className="list-disc ml-6 mt-2">
              <li>Architected Redshift warehouse & automated ETL pipelines.</li>
              <li>Developed Python scripts for data validation in Snowflake.</li>
              <li>Worked with Step Functions, Lambda, and Kinesis for real-time processing.</li>
              <li>🏆 Rising Star Award (2023) for project excellence.</li>
            </ul>
          </div>
          <div>
            <h3 className="text-xl font-bold">AWS Data Engineer – Exponentia.ai</h3>
            <p className="text-gray-700">Feb 2024 – May 2024</p>
            <ul className="list-disc ml-6 mt-2">
              <li>Designed ETL pipelines for insurance domain clients.</li>
              <li>Worked with AWS RDS, S3, SageMaker, and PostgreSQL.</li>
            </ul>
          </div>
          <div>
            <h3 className="text-xl font-bold">People Analytics Intern – Capgemini Chrysalis</h3>
            <p className="text-gray-700">Apr 2025 – Jun 2025</p>
            <p>Applied analytics to HR and people data for insights.</p>
          </div>
        </div>
      </motion.section>

      {/* Skills */}
      <motion.section
        id="skills"
        className="p-20 text-center"
        variants={sectionVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <h2 className="text-3xl font-semibold">Skills</h2>
        <div className="flex flex-wrap justify-center mt-6 gap-4">
          {["AWS Glue", "Redshift", "Athena", "S3", "Lambda", "SQL", "Python", "Spark", "Snowflake"].map((skill) => (
            <span key={skill} className="px-4 py-2 bg-blue-100 text-blue-800 rounded-full shadow">
              {skill}
            </span>
          ))}
        </div>
      </motion.section>

      {/* Projects */}
      <motion.section
        id="projects"
        className="p-20 bg-gray-100"
        variants={sectionVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <h2 className="text-3xl font-semibold text-center mb-10">Projects & Knowledge Base</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-5xl mx-auto">
          {repos.map((repo) => (
            <div
              key={repo.id}
              className="p-6 bg-white rounded-xl shadow hover:shadow-lg transition"
            >
              <h3 className="text-xl font-bold">{repo.name}</h3>
              <p className="mt-2 text-gray-700">
                {repo.description || "This repository has no description."}
              </p>
              <p className="mt-2 text-sm text-gray-500">
                ⭐ {repo.stargazers_count} | 🍴 {repo.forks_count}
              </p>
              <a
                href={repo.html_url}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-4 inline-block px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                View on GitHub →
              </a>
            </div>
          ))}
        </div>
      </motion.section>

      {/* Contact */}
      <motion.section
        id="contact"
        className="p-20 text-center"
        variants={sectionVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <h2 className="text-3xl font-semibold">Let’s Connect</h2>
        <p className="mt-4">📧 your.email@example.com</p>
        <div className="mt-6 space-x-4">
          <a href="https://linkedin.com/in/tejaspatil4873" className="text-blue-600 hover:underline">LinkedIn</a>
          <a href="https://github.com/tejaspatil4873" className="text-blue-600 hover:underline">GitHub</a>
        </div>
      </motion.section>
    </div>
  );
};

export default PortfolioApp;
