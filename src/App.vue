<script lang="ts">
import { RouterLink, RouterView } from "vue-router";
import { defineComponent } from 'vue'
import router from "./router";

export default defineComponent({
  setup() {
    return {
      base: import.meta.env.BASE_URL,
      options: [
        {
          label: "Secret Santa",
          key: "/projects"
        },
        {
          label: "Animal Voice App",
          key: "/projects"
        }
      ],
      handleSelect(key: string) {
        router.push(key);
      }
    }
  }
})
</script>
<template>
  <n-layout style="background: #011627" position="absolute" :native-scrollbar="false">
    <n-layout-header
      mode="horizontal"
      style="background: #e4572e; padding-left: 10px"
    >
      <n-grid cols="1 xs:1 s:1 m:1 l:5 xl:6 2xl:6" responsive="screen" style="height: 50px; width: 100%">
        <n-gi>
          <n-space justify="space-around" align="center" style="height: 100%">
            <router-link class="topNavigation" to="/">Home</router-link>
            <router-link class="topNavigation" to="/blogs">Blogs</router-link>
            <n-dropdown :show-arrow="true" size="small" placement="bottom-start" trigger="hover" :options="options" @select="handleSelect"><h3 class="dropdown" @click="$router.push('/projects')">Projects</h3></n-dropdown>
            <a href="../assets/ChristopherGSmith3975_Resume.pdf" download
              >My Resume</a
            >
          </n-space>
        </n-gi>
      </n-grid>
    </n-layout-header>
    <n-layout>
      <n-space justify="center" style="background: #011627; min-height: 750px">
        <RouterView />
      </n-space>
    </n-layout>
    <n-layout-footer style="background: #e07a5f">
      <n-grid
        style="min-height: 100px; width: 100%; padding-top: 20px"
        cols="3 xs:.5 s:1 m:2 l:2 xl:2 2xl:2" responsive="screen"
        :x-gap="100"
      >
        <n-gi>
          <n-space vertical align="center">
            <n-h3>Address: Mansfield, TX, 76063</n-h3>
            <n-h3>Phone: (682)-558-3975</n-h3>
          </n-space>
        </n-gi>
        <n-gi>
          <n-space vertical align="center">
            <n-h3>Email: christophergsmith3975@gmail.com</n-h3>
            <n-h3
              >LinkedIn Profile:
              <a
                href="https://www.linkedin.com/in/christophergsmith3975"
                target="_blank"
                >www.linkedin.com/in/christophergsmith3975</a
              ></n-h3
            >
          </n-space>
        </n-gi>
      </n-grid>
    </n-layout-footer>
  </n-layout>
</template>

<style scoped>
</style>
