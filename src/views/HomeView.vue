<script setup lang="ts">
</script>

<template>
  <n-space vertical style="margin-top: 50px; margin-bottom: 50px">
    <n-grid :cols="2">
      <n-gi style="height: 400px">
        <n-space vertical align="center"
          ><n-h1 style="color: #e4572e">Christopher Smith</n-h1>
          <n-avatar round src="src/assets/professionalpic.jpg" :size="300"
        /></n-space>
      </n-gi>
      <n-gi style="height: 400px"
        ><n-space vertical justify="space-around" style="margin-top: 80px"
          ><n-h2 style="color: white; width: 600px"
            >I'm an undergradute student studying Computer Science at The
            University of Texas at Arlington.</n-h2
          >
          <n-h2 style="color: white; width: 600px"
            >I work part-time at Ayoka Systems as an intern for Web/Application
            Development.</n-h2
          >
          <n-h2 style="color: white; width: 600px">
            My interests are in AI and Web Application Development.
          </n-h2>
        </n-space>
      </n-gi>
    </n-grid>
    <n-h1 style="color: #e4572e; padding-left: 20px">Education</n-h1>
    <n-grid
      style="width: 100%; padding-left: 60px; padding-right: 60px"
      :cols="2"
      item-responsive
      :x-gap="100"
    >
      <n-gi>
        <n-space vertical>
          <n-h2 style="color: white">MAY 2023</n-h2>
          <n-h2 style="color: #e07a5f"
            >Bachelor's in Computer Science, The University of Texas at
            Arlington</n-h2
          >
          <n-h3 style="color: white"
            >Currently attending the University of Texas at Arlington as a 4th
            year student. I have held a consistent GPA of 3.4. My classes this
            semester are currently Data Mining, Artifiical Intelligence, and
            Robotics.</n-h3
          >
        </n-space>
      </n-gi>
      <n-gi>
        <n-space vertical>
          <n-h2 style="color: white">MAY 2019</n-h2>
          <n-h2 style="color: #e07a5f"
            >High School Diploma, Mansfield High School</n-h2
          >
          <n-h3 style="color: white"
            >I gained experience from classes in different fields such as:
            Object-Oriented Programming, Network and Computer Maintenance, and
            Drone operation.</n-h3
          >
        </n-space>
      </n-gi>
    </n-grid>
    <n-h1 style="color: #e4572e; padding-left: 20px">Experience</n-h1>
    <n-grid
      style="width: 100%; padding-left: 60px; padding-right: 60px"
      :cols="2"
      item-responsive
      :x-gap="100"
    >
      <n-gi>
        <n-space vertical>
          <n-h3 style="color: white">2022 - Current</n-h3>
          <n-h3 style="color: #e07a5f"
            >Intern for Web/App Development, Ayoka Systems, Arlington, TX</n-h3
          >
          <n-h3 style="color: white">Front-end/Back-end development</n-h3>
          <n-h3 style="color: white"
            >Working with Vue 2 and Asp.Net frameworks</n-h3
          >
        </n-space>
      </n-gi>
      <n-gi>
        <n-space vertical>
          <n-h3 style="color: white">2018 - 2019</n-h3>
          <n-h3 style="color: #e07a5f"
            >Intern, MISD IT Department, Mansfield, TX</n-h3
          >
          <n-h3 style="color: white"
            >Maintaining and troubeshooting servers and desktops</n-h3
          >
          <n-h3 style="color: white">Professional client communication</n-h3>
        </n-space>
      </n-gi>
    </n-grid>
    <n-h1 style="color: #e4572e; padding-left: 20px">Skills</n-h1>
    <n-grid
      style="width: 100%; padding-left: 60px; padding-right: 60px"
      :cols="2"
      item-responsive
      :x-gap="100"
    >
      <n-gi>
        <n-space vertical>
          <n-h3 style="color: white">JavaScript</n-h3>
          <n-h3 style="color: white">Html5/CSS</n-h3>
          <n-h3 style="color: white">Web/App Development</n-h3>
        </n-space>
      </n-gi>
      <n-gi>
        <n-space vertical>
          <n-h3 style="color: white">Vue 2/3</n-h3>
          <n-h3 style="color: white">Python</n-h3>
          <n-h3 style="color: white">SQL</n-h3>
        </n-space>
      </n-gi>
    </n-grid>
    <n-h1 style="color: #e4572e; padding-left: 20px">Projects</n-h1>
    <n-grid
      style="width: 100%; padding-left: 60px; padding-right: 60px"
      :cols="2"
      item-responsive
      :x-gap="100"
    >
      <n-gi>
        <n-space vertical>
          <n-h3 style="color: #e07a5f">Secret Santa</n-h3>
          <n-h3 style="color: white"
            >In this project my team designed a website that would automate the
            process and preserve the secrecy of the traditional "Secret Santa"
            practiced by families. We utilized Vue as the framework for this
            project so JavaScript, CSS, and HTML could work in harmony. We used
            GitHub for our repository so we could work in parallel and
            synchronize when needed.</n-h3
          >
        </n-space>
      </n-gi>
      <n-gi>
        <n-space vertical>
          <n-h3 style="color: #e07a5f">Animal Voice App</n-h3>
          <n-h3 style="color: white; width: 650px"
            >In this project my team designed an app that would take in audio
            files and decipher what was recorded. We used Andriod Studio and its
            GUI system to design the layout of the app. Then we used Firebase to
            create user accounts with data such as a username and password. To
            decipher the animal, we used python and implemented machine learning
            from articles given.</n-h3
          >
        </n-space>
      </n-gi>
    </n-grid>
  </n-space>
</template>
