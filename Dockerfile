FROM python:3.6.5 as buildml
WORKDIR /home/
RUN apt update && git clone https://github.com/SMYALTAMASH/sunbird-ml-workbench
COPY app.go .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o app .

FROM python:3.6.5-apline  
RUN apk update
WORKDIR /home/
COPY --from=buildml /go/src/github.com/alexellis/href-counter/app .
CMD ["./app"]  