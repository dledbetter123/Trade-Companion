// import logo from './logo.svg';
import {Line} from 'react-chartjs-2';
import { useState } from 'react'
import axios from "axios";
import './App.css';
import Plot from 'react-plotly.js';

function App() {
  
  // var dateRange = function(startDate, endDate, steps = 1) {
  //   const dateArray = [];
  //   let currentDate = new Date(startDate);
  
  //   while (currentDate <= new Date(endDate)) {
  //     dateArray.push(new Date(currentDate));
  //     // Use UTC date to prevent problems with time zones and DST
  //     currentDate.setUTCDate(currentDate.getUTCDate() + steps);
  //   }
  
  //   return dateArray;
  // };
  
  // const dates = dateRange('2020-09-27', '2020-10-28');
  // console.log(dates);

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
          Xb: res.bplot_x,
          Xp: res.pplot_x,
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
              x: predictData.Xb,
              y: predictData.all,
              type: 'scatter',
              mode: 'lines',
              marker: {color: 'red'},
              name: "APPL"
            },
            {
              x: predictData.Xp,
              y: predictData.predict,
              type: 'scatter',
              mode: 'lines',
              marker: {color: 'green'},
              name: "APPL "+String(predictData.Xp.length)
            }
          ]}
          layout={ {width: 1000, height: 480, title: 'APPL', paper_bgcolor: "black", plot_bgcolor:"black"} }
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
