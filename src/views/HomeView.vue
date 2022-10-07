<script lang="ts">
</script>

<template>
  <n-space vertical style="padding: 50px 100px 50px 100px">
    <n-grid style="width: 100%" cols="2 xs:1 s:1 m:1 l:2 xl:2 2xl:2" responsive="screen">
      <n-gi>
        <n-space vertical align="center"
          ><n-h1 style="color: #e4572e">Christopher Smith</n-h1>
          <n-avatar round src="assets/ResumeProfilePic.jpg" :size="300"
        /></n-space>
      </n-gi>
      <n-gi
        ><n-space vertical justify="space-around" style="margin-top: 80px"
          ><n-h2 style="color: white;"
            >I'm an undergraduate student studying Computer Science at The
            University of Texas at Arlington.</n-h2
          >
          <n-h2 style="color: white;"
            >I work part-time at Ayoka Systems as an intern for Web/Application
            Development.</n-h2
          >
          <n-h2 style="color: white;">
            My interests are in AI and Web Application Development.
          </n-h2>
        </n-space>
      </n-gi>
    </n-grid>
    <n-divider>
    <n-h1 style="color: #e4572e">Education</n-h1> </n-divider>
    <n-grid cols="2 xs:1 s:1 m:1 l:2 xl:2 2xl:2" responsive="screen" :x-gap="100">
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
            semester are currently Data Mining, Artificial Intelligence, and
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
    <n-divider>
    <n-h1 style="color: #e4572e">Experience</n-h1></n-divider>
    <n-grid cols="2 xs:1 s:1 m:1 l:2 xl:2 2xl:2" responsive="screen" :x-gap="100">
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
            >Maintaining and troubleshooting servers and desktops</n-h3
          >
          <n-h3 style="color: white">Professional client communication</n-h3>
        </n-space>
      </n-gi>
    </n-grid>
    <n-divider >
    <n-h1 style="color: #e4572e">Skills</n-h1></n-divider>
    <n-grid cols="2 xs:1 s:1 m:1 l:2 xl:2 2xl:2" responsive="screen" :x-gap="100">
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
    <n-divider >
    <n-h1 style="color: #e4572e">Projects</n-h1></n-divider>
    <n-grid cols="2 xs:1 s:1 m:1 l:2 xl:2 2xl:2" responsive="screen" :x-gap="100">
      <n-gi>
        <n-space vertical>
          <n-h3 id="project" style="color: #e07a5f; cursor: pointer" @click="$router.push('/projects')">Secret Santa</n-h3>
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
          <n-h3 id="project" style="color: #e07a5f; cursor: pointer" @click="$router.push('/projects')">Animal Voice App</n-h3>
          <n-h3 style="color: white;"
            >In this project my team designed an app that would take in audio
            files and decipher what was recorded. We used Android Studio and its
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
