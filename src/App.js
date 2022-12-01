// import logo from './logo.svg';
import {Line} from 'react-chartjs-2';
import { useState } from 'react'
import axios from "axios";
import './App.css';
import Plot from 'react-plotly.js';

function App() {

  const [profileData, setProfileData] = useState(null)
  const [predictData, setPredictData] = useState(null)

  function getProfileData() {
    axios({
      method: "GET",
      url:"/profile",
    })
    .then((response) => {
      const res =response.data
      setProfileData(({
        profile_name: res.name,
        about_me: res.about}))
    }).catch((error) => {
      if (error.response) {
        console.log(error.response)
        console.log(error.response.status)
        console.log(error.response.headers)
        }
    })}

    function getPredictData() {
      axios({
        method: "GET",
        url:"/predict",
      })
      .then((response) => {
        const res =response.data
        setPredictData(({
          base: res.base,
          predict: res.prediction,
          size: res.size,
          all: res.all
        }))
      }).catch((error) => {
        if (error.response) {
          console.log(error.response)
          console.log(error.response.status)
          console.log(error.response.headers)
          }
      })}


  return (
    <div className="App">
      <header className="App-header">
      <button onClick={getPredictData}>Display Prediction</button>{
        predictData && <div>
        <Plot
          data={[
            {
              x: [...Array(predictData.size).keys()],
              y: predictData.all,
              type: 'scatter',
              mode: 'lines',
              marker: {color: 'red'},
            },
          ]}
          layout={ {width: 1000, height: 480, title: 'Plot'} }
        />
      </div>
      }

        <p>To get your profile details: </p><button onClick={getProfileData}>Click me</button>
        {profileData && <div>
              <p>Profile name: {profileData.profile_name}</p>
              <p>About me: {profileData.about_me}</p>
            </div>
        }


      </header>
    </div>
  );
}

export default App;